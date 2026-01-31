from loader import load_all_pdf
from chunking import chunk_doc
from vectorstore import build_and_save

docs = load_all_pdf("data/raw_docs")
chunks = chunk_doc(docs)

build_and_save(chunks)

print("FAISS index built and saved successfully.")
