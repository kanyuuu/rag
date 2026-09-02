# build_index.py

import os
import torch
import json
import faiss
from dotenv import load_dotenv

# --- 更新后的导包 ---
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Sequence, Optional, Iterator, Tuple
from langchain_core.stores import BaseStore
from langchain_core.documents import Document

# --- 配置 ---
os.environ['HF_HOME'] = './my_models_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
load_dotenv()

PDF_PATH = "./政务服务数据.pdf"
EMBEDDING_MODEL_NAME_OR_PATH = "BAAI/bge-small-zh-v1.5"
FAISS_DB_PATH = "./faiss_index"
METADATA_FILE_NAME = "documents_metadata.json"

PARENT_CHUNK_SIZE = 800  
CHILD_CHUNK_SIZE = 200   
PARENT_STORE_PATH = "./parent_docs_store"  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def create_parent_child_index(pdf_path: str, embeddings_model, db_path: str, store_path: str):
    print(f"正在使用 PDFPlumber 加载 PDF 文件: {pdf_path}")
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    print(f"原始文档页数: {len(documents)}")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=50)

    if not os.path.exists(store_path):
        os.makedirs(store_path)
    # 换成我们自己手搓的引擎
    store = SimpleLocalDocStore(store_path)

    print("正在初始化空 FAISS 库...")
    dummy_embed = embeddings_model.embed_query("test")
    index = faiss.IndexFlatL2(len(dummy_embed))
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("正在进行父子块切分并存入数据库 (这可能需要一些时间)...")
    retriever.add_documents(documents)

    vectorstore.save_local(db_path)
    print(f"子文档 FAISS 向量库已保存至: {db_path}")
    print(f"父文档文本已自动持久化至: {store_path}")

def get_embeddings_model(model_name_or_path: str) -> HuggingFaceEmbeddings:
    print(f"正在加载嵌入模型: {model_name_or_path} (device: {DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={'device': DEVICE}
    )
    print("嵌入模型加载完成。")
    return embeddings

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"错误：PDF 文件未找到，请确保 '{PDF_PATH}' 存在。")
    else:
        try:
            embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)
            create_parent_child_index(PDF_PATH, embeddings_model, FAISS_DB_PATH, PARENT_STORE_PATH)
            print("\n索引构建完成！")
        except Exception as e:
            print(f"\n构建过程中发生错误: {e}")