# app.py

import os
import torch
import requests
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# LangChain 和向量数据库相关的导入
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import Sequence, Optional, Iterator, Tuple
from langchain_core.stores import BaseStore

# --- 1. 初始化和配置 ---
print("正在初始化 FastAPI 应用和 RAG 系统...")

# 加载环境变量 (SILICONFLOW_API_KEY)
load_dotenv()
os.environ['HF_HOME'] = './my_models_cache'  # 设置HuggingFace缓存目录
# 配置 Hugging Face 国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME_OR_PATH = "BAAI/bge-small-zh-v1.5" 
#EMBEDDING_MODEL_NAME_OR_PATH = "./bge-small-zh-v1.5"
FAISS_DB_PATH = "./faiss_index"
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# Reranker 和 LLM 模型配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # 使用一个高效的指令模型
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

class SimpleLocalDocStore(BaseStore[str, Document]):
    """自定义本地文档存储，绕过 Langchain 环境坑，直接以 JSON 格式持久化父文档"""
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

# --- 2. 加载模型和数据 (在应用启动时执行一次) ---
# 检查API密钥
if not SILICONFLOW_API_KEY:
    raise ValueError("错误：环境变量 SILICONFLOW_API_KEY 未设置！")

# 检查FAISS索引是否存在
if not os.path.exists(FAISS_DB_PATH):
    raise FileNotFoundError(f"错误：FAISS 索引目录 '{FAISS_DB_PATH}' 未找到。请先运行 build_index.py 来创建索引。")

# 加载嵌入模型
print(f"正在加载嵌入模型: {EMBEDDING_MODEL_NAME_OR_PATH} 到设备: {DEVICE}")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME_OR_PATH,
    model_kwargs={'device': DEVICE}
)

# 加载FAISS向量数据库
print(f"正在从 '{FAISS_DB_PATH}' 加载FAISS数据库...")
faiss_db = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True
)

# 创建检索器
# 加载用于存放父文档的本地存储
PARENT_STORE_PATH = "./parent_docs_store"
store = SimpleLocalDocStore(PARENT_STORE_PATH)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# 重建 ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=faiss_db,
    docstore=store,
    child_splitter=child_splitter, # 检索阶段不需要重新切分
    parent_splitter=parent_splitter,
    search_kwargs={"k": 10} # 检索10个子块，进而映射回它们所属的父块
)
print("父子块 RAG 系统初始化完成，准备好接收请求。")


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


# --- 4. 辅助函数 (Reranker 和 LLM 调用) ---

def rerank_documents(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """使用 SiliconFlow API 对文档进行重排"""
    doc_contents = [doc.page_content for doc in docs]
    payload = {"model": RERANKER_MODEL, "query": query, "documents": doc_contents}
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(f"{SILICONFLOW_API_BASE}/rerank", json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        rerank_results = response.json().get("results", [])

        # 将rerank结果与原始文档关联并排序
        reranked_items = [{"document": docs[res['index']], "score": res['relevance_score']} for res in rerank_results]
        reranked_items.sort(key=lambda x: x['score'], reverse=True)

        return [item['document'] for item in reranked_items[:top_n]]
    except requests.RequestException as e:
        print(f"Reranker API 调用失败: {e}")
        return docs[:top_n]  # 如果rerank失败，返回初始检索的前N个文档作为备用


def generate_answer(query: str, context_docs: list[Document], history: List[Dict[str, str]] = None) -> str:
    """使用 SiliconFlow API 和重排后的文档生成多轮对话答案"""
    if history is None:
        history = []
        
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # 1. 构建 System Prompt，赋予角色并注入检索到的上下文
    system_prompt = f"""你是一个专业的政务服务助手。
请根据以下提供的上下文信息，准确、详细地回答用户的当前问题。
如果上下文信息不足以回答，请明确说明“根据现有信息无法回答该问题”。

上下文信息:
---
{context}
---"""

    # 2. 组装 messages 列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 3. 追加历史对话 (为了防止 token 超出限制，建议只保留最近的 3-5 轮对话)
    # 假设我们保留最近的 6 条消息 (3轮交互)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # 4. 追加当前用户的最新问题
    messages.append({"role": "user", "content": query})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.1, # 保持低温度以保证政务回答的准确性
    }
    
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(f"{SILICONFLOW_API_BASE}/chat/completions", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"LLM API 调用失败: {e}")
        return "抱歉，调用大语言模型生成答案时发生网络错误。"
    except (KeyError, IndexError) as e:
        print(f"LLM API 响应格式错误: {e}, 响应内容: {response.text}")
        return "抱歉，解析大语言模型返回的答案时发生错误。"

# --- 5. FastAPI 应用和 API 路由 ---
app = FastAPI(title="RAG API", description="基于 FastAPI 的 RAG 问答系统")


@app.post("/rag_query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    接收用户问题，执行 RAG+Rerank 流程，并返回LLM生成的答案。
    """
    question = request.question

    if not question:
        raise HTTPException(status_code=400, detail="请求体中必须包含 'question' 字段")

    print(f"\n收到新请求: {question}")

    try:
        # 步骤 1: 初始检索
        print("步骤 1: 正在从FAISS进行初始检索...")
        initial_docs = retriever.invoke(question)
        print(f"  - 检索到 {len(initial_docs)} 篇初始文档。")

        # 步骤 2: 文档重排
        print("步骤 2: 正在使用Reranker进行重排...")
        reranked_docs = rerank_documents(question, initial_docs, top_n=3)
        print(f"  - 重排后保留 {len(reranked_docs)} 篇文档。")

        # 步骤 3: 生成答案
        print("步骤 3: 正在调用LLM生成最终答案...")
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
        answer = generate_answer(question, reranked_docs, history_dicts)
        print(f"  - LLM生成答案完成。")

        # 准备返回的源文档信息
        source_documents = [SourceDocument(
            content=doc.page_content,
            metadata=doc.metadata
        ) for doc in reranked_docs]

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


# --- 6. 启动应用 ---
if __name__ == '__main__':
    import uvicorn
    # 在生产环境中，应使用 Gunicorn 或其他 ASGI 服务器，而不是 uvicorn 的开发服务器
    uvicorn.run(app, host='0.0.0.0', port=5000)