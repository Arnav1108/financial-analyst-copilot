import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "data/faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_and_save(chunks):
    embedding = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

def load_vectorstore():
    embedding = get_embeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

def index_exists():
    return os.path.exists(INDEX_PATH)
