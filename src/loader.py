import os
from langchain_community.document_loaders import PyPDFLoader

def load_all_pdf(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path,file)
            loader = PyPDFLoader(full_path)
            doc = loader.load()

            for d in doc:
                d.metadata["doc_name"] = file
                documents.append(d)

    return documents