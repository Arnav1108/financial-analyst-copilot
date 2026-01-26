from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def building_vector(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embedding)


