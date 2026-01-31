from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

ANSWER_PROMPT  = PromptTemplate(
    input_variables=["context","question"],
    template="""
You are a financial research assistant.

Use ONLY the context below to answer the question.
Cite evidence by referencing the source metadata.

If the answer is not clearly supported, say "I don't know".

Context:
{context}

Question:
{question}

Answer with citations:
"""
)

COMPARE_PROMPT = PromptTemplate(
    input_variables=["context", "question","chat_history"],
    template="""
You are a financial research assistant.

Conversation so far:
{chat_history}

Using ONLY the context below:

You MUST combine repeated causes into ONE statement.
If multiple sources say the same thing, mention it once with multiple citations.
Never repeat the same explanation in different wording.

• Merge repeated information
• Do NOT restate the same cause multiple times
• Summarize unique drivers clearly

Identify:
- what increased
- what decreased
- the difference
- the main causes

Your answer should be short, non-repetitive, and compressed like an analyst summary.

If evidence is insufficient, say "I don't know".

Context:
{context}

Question:
{question}

Give a concise synthesized answer with citations.
"""
)

def format_doc(docs):
    return "\n\n".join(
        f"[Document: {d.metadata.get('doc_name')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )
