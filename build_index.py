# build_index.py

import os
import torch
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 配置 ---
# 设置缓存目录
os.environ['HF_HOME'] = './my_models_cache'
# 配置 Hugging Face 国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 加载环境变量
load_dotenv()

# 文件和模型路径配置
PDF_PATH = "./政务服务数据.pdf"
EMBEDDING_MODEL_NAME_OR_PATH = "BAAI/bge-small-zh-v1.5"
FAISS_DB_PATH = "./faiss_index" # 简化了路径名
METADATA_FILE_NAME = "documents_metadata.json"

# 切分参数
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# 计算设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """加载 PDF 文档并进行文本切分。"""
    print(f"正在加载 PDF 文件: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"原始文档页数: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档切分完成，生成 {len(chunks)} 个文本块。")
    return chunks

def get_embeddings_model(model_name_or_path: str) -> HuggingFaceEmbeddings:
    """获取嵌入模型。"""
    print(f"正在加载嵌入模型: {model_name_or_path} (device: {DEVICE})")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={'device': DEVICE}
    )
    print("嵌入模型加载完成。")
    return embeddings

def create_and_save_faiss_db(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, db_path: str):
    """创建 FAISS 向量数据库并保存到本地。"""
    print("正在创建 FAISS 向量数据库...")
    faiss_db = FAISS.from_documents(chunks, embeddings_model)
    print(f"FAISS 向量数据库创建完成。正在保存到: {db_path}")
    faiss_db.save_local(db_path)
    print("FAISS 向量数据库保存成功。")

def create_and_save_metadata(chunks: list[Document], output_dir: str, metadata_file_name: str):
    """创建并保存文档块的元数据。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadata_list = [{"chunk_id": i, "content": chunk.page_content, "metadata": chunk.metadata} for i, chunk in enumerate(chunks)]
    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print(f"元数据已保存到：{metadata_file_path}")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"错误：PDF 文件未找到，请确保 '{PDF_PATH}' 存在。")
    else:
        try:
            # 1. 加载并切分 PDF
            document_chunks = load_and_split_pdf(PDF_PATH)
            # 2. 初始化嵌入模型
            embeddings_model = get_embeddings_model(EMBEDDING_MODEL_NAME_OR_PATH)
            # 3. 创建并保存 FAISS 向量数据库
            create_and_save_faiss_db(document_chunks, embeddings_model, FAISS_DB_PATH)
            # 4. 创建并保存元数据
            create_and_save_metadata(document_chunks, FAISS_DB_PATH, METADATA_FILE_NAME)
            print("\n索引和元数据构建完成！")
        except Exception as e:
            print(f"\n构建过程中发生错误: {e}")