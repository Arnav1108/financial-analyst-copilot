from dotenv import load_dotenv
load_dotenv()

from src.chunking import chunk_doc
from src.vectorstore import building_vector
from src.loader import load_pdf

docs = load_pdf("data/raw_docs/saudi-aramco-q3.pdf")
chunks = chunk_doc(docs)

print("Chunks" , len(chunks))

vector_store = building_vector(chunks)
retriver = vector_store.as_retriever(search_kwargs={"k":3})

results = retriver.invoke("what drove the revenue growth?")

for r in results:
    print("----")
    print(r.page_content[:300])


