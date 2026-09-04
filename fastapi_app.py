# fastapi_app.py

import os
import torch
import requests
import json
import pickle
import jieba
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional, Iterator, Tuple

# LangChain 和向量数据库相关的导入
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import BaseStore

# --- 0. 手搓本地文档存储 (绕过 Langchain 环境坑) ---
class SimpleLocalDocStore(BaseStore[str, Document]):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        
    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        docs = []
        for key in keys:
            file_path = os.path.join(self.path, f"{key}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    docs.append(Document(page_content=data['page_content'], metadata=data.get('metadata', {})))
            else:
                docs.append(None)
        return docs
        
    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        for key, doc in key_value_pairs:
            file_path = os.path.join(self.path, f"{key}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({'page_content': doc.page_content, 'metadata': doc.metadata}, f, ensure_ascii=False)
                
    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            file_path = os.path.join(self.path, f"{key}.json")
            if os.path.exists(file_path):
                os.remove(file_path)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        if not os.path.exists(self.path):
            return
        for filename in os.listdir(self.path):
            if filename.endswith(".json"):
                key = filename[:-5]
                if prefix is None or key.startswith(prefix):
                    yield key

# --- 1. 初始化和配置 ---
print("正在初始化 FastAPI 应用和 RAG 系统...")

load_dotenv()
os.environ['HF_HOME'] = './my_models_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME_OR_PATH = "BAAI/bge-small-zh-v1.5" 
FAISS_DB_PATH = "./faiss_index"
BM25_INDEX_PATH = "./bm25_index.pkl"
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# --- 2. 加载模型和数据 ---
if not SILICONFLOW_API_KEY:
    raise ValueError("错误：环境变量 SILICONFLOW_API_KEY 未设置！")

# 加载嵌入模型
print(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME_OR_PATH} 到设备: {DEVICE}")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME_OR_PATH,
    model_kwargs={'device': DEVICE}
)

# 2.1 加载 FAISS 向量数据库
if not os.path.exists(FAISS_DB_PATH):
    raise FileNotFoundError(f"错误：FAISS 索引目录 '{FAISS_DB_PATH}' 未找到。")
print(f"正在从 '{FAISS_DB_PATH}' 加载FAISS数据库...")
faiss_db = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True
)

# 2.2 加载 BM25 索引
def jieba_tokenize(text: str):
    return jieba.lcut(text)

if not os.path.exists(BM25_INDEX_PATH):
    print(f"警告：BM25 索引文件 '{BM25_INDEX_PATH}' 未找到，请确保已更新 build_index.py 并重新建库。")
    bm25_retriever = None
else:
    print(f"正在从 '{BM25_INDEX_PATH}' 加载BM25索引...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

# 2.3 构建父子块检索器
PARENT_STORE_PATH = "./parent_docs_store"
store = SimpleLocalDocStore(PARENT_STORE_PATH)

# 实例化切分器以骗过 Pydantic 校验（参数与建库时保持一致）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

retriever = ParentDocumentRetriever(
    vectorstore=faiss_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10} 
)
print("系统初始化完成，准备好接收请求。")


# --- 3. Pydantic 模型定义 ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    source_documents: List[SourceDocument]

class HealthResponse(BaseModel):
    status: str
    message: str


# --- 4. 核心功能函数 (新增 Query Rewrite 与 RRF 双路召回) ---

def rewrite_query(original_query: str, history: List[Dict[str, str]]) -> str:
    """利用 LLM 根据历史对话重写当前问题"""
    if not history:
        return original_query
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    prompt = f"""你是一个查询重写专家。请根据以下历史对话，将用户的最新问题重写为一个独立、完整、意图明确的查询请求。
如果不需要重写，请直接输出原问题。只需输出重写后的问题，不要输出任何额外解释。

历史对话:
{history_text}

用户最新问题: {original_query}
重写后的问题:"""
    
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(f"{SILICONFLOW_API_BASE}/chat/completions", json=payload, headers=headers, timeout=20)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"查询重写失败，退回原问题: {e}")
        return original_query

def hybrid_search(query: str, retriever_faiss, retriever_bm25, top_k=10):
    """基于 RRF（倒数排名融合）算法的自定义双路召回"""
    docs_faiss = retriever_faiss.invoke(query)
    
    # 如果 BM25 没有成功加载，退化为纯 FAISS 召回
    if not retriever_bm25:
        return docs_faiss[:top_k]
        
    docs_bm25 = retriever_bm25.invoke(query)
    
    fused_scores = {}
    doc_map = {}
    
    def apply_rrf(docs, weight):
        for rank, doc in enumerate(docs):
            doc_id = doc.page_content 
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += weight / (rank + 60)
            
    apply_rrf(docs_faiss, weight=0.5)
    apply_rrf(docs_bm25, weight=0.5)
    
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[k] for k, v in sorted_results[:top_k]]

def rerank_documents(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """使用 SiliconFlow API 对文档进行重排"""
    if not docs:
        return []
    doc_contents = [doc.page_content for doc in docs]
    payload = {"model": RERANKER_MODEL, "query": query, "documents": doc_contents}
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(f"{SILICONFLOW_API_BASE}/rerank", json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        rerank_results = response.json().get("results", [])

        reranked_items = [{"document": docs[res['index']], "score": res['relevance_score']} for res in rerank_results]
        reranked_items.sort(key=lambda x: x['score'], reverse=True)

        return [item['document'] for item in reranked_items[:top_n]]
    except requests.RequestException as e:
        print(f"Reranker API 调用失败: {e}")
        return docs[:top_n] 

def generate_answer(query: str, context_docs: list[Document], history: List[Dict[str, str]] = None) -> str:
    """使用 LLM 生成最终答案"""
    if history is None:
        history = []
        
    context = "\n\n".join([doc.page_content for doc in context_docs])
    system_prompt = f"""你是一个专业的政务服务助手。
请根据以下提供的上下文信息，准确、详细地回答用户的当前问题。
如果上下文信息不足以回答，请明确说明“根据现有信息无法回答该问题”。

上下文信息:
---
{context}
---"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1, 
    }
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(f"{SILICONFLOW_API_BASE}/chat/completions", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM API 调用失败: {e}")
        return "抱歉，调用大语言模型生成答案时发生网络错误。"


# --- 5. FastAPI 路由 ---
app = FastAPI(title="Agentic RAG API", description="基于 FastAPI 的智能政务 RAG 系统")

@app.post("/rag_query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="请求体中必须包含 'question' 字段")

    print(f"\n[新请求] 原始问题: {question}")
    try:
        # 步骤 1: 查询重写 (Query Rewrite)
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
        rewritten_query = rewrite_query(question, history_dicts)
        if rewritten_query != question:
            print(f"  -> 重写为: {rewritten_query}")

        # 步骤 2: 双路召回 (Hybrid Search)
        print("步骤 2: 执行 BM25 + FAISS 双路召回 (RRF 融合)...")
        initial_docs = hybrid_search(rewritten_query, retriever, bm25_retriever, top_k=10)
        print(f"  - 召回了 {len(initial_docs)} 篇初始文档。")

        # 步骤 3: 文档重排 (Rerank)
        print("步骤 3: 使用 BGE-Reranker 进行精排...")
        reranked_docs = rerank_documents(rewritten_query, initial_docs, top_n=3)
        print(f"  - 精排后保留 {len(reranked_docs)} 篇文档。")

        # 步骤 4: LLM 生成答案
        print("步骤 4: 调用 LLM 生成最终答案...")
        answer = generate_answer(rewritten_query, reranked_docs, history_dicts)
        print(f"  - 回答生成完毕。")

        source_documents = [SourceDocument(content=doc.page_content, metadata=doc.metadata) for doc in reranked_docs]

        return QueryResponse(
            success=True,
            question=question,
            answer=answer,
            source_documents=source_documents
        )
    except Exception as e:
        print(f"处理请求时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="RAG API 服务正在运行")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)