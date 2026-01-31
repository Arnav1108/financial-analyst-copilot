from dotenv import load_dotenv
load_dotenv()


from src.llm import get_llm
from src.vectorstore import load_vectorstore
from src.qa import ANSWER_PROMPT,format_doc,COMPARE_PROMPT
from src.confidence import compute_confidence, confidence_label

vector_store = load_vectorstore()
llm = get_llm()
chat_his = ""

while True:
    ques = input("\n Ask your ques (or exit): ")

    results_with_score = vector_store.similarity_search_with_score(
    ques,
    k=5
    )

    results = [doc for doc, _ in results_with_score]
    distances = [score for _, score in results_with_score]

    context = format_doc(results)

    prompt = COMPARE_PROMPT.format(
        context=context,
        question=ques,
        chat_history=chat_his
    )

    response = llm.invoke(prompt)

    chat_his += f"\nUser:{ques}\nAI:{response.content}\n"

    score = compute_confidence(distances)
    label = confidence_label(score)

    print(response.content)
    print(f"\nConfidence: {score}/10 — {label}")