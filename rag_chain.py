# rag_pipeline.py - Option C (Corrected, real usable RAG)
import os
import re
from langchain_core.runnables import RunnableLambda
from azure_llm_client import CustomFarmAzureLLM 

from vector_store import get_vector_database, store_chunks_in_chromadb, CHROMA_DB_PATH
from document_ingestion import load_documents, semantic_chunk_free_local, DATA_PATH


os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "PATH_TO_LLM FARM"

AZURE_DEPLOYMENT_NAME = ""
AZURE_API_VERSION = ""


def initialize_llm():
    print("🔄 Connecting to Farm Azure GPT...")
    llm = CustomFarmAzureLLM(
        endpoint="",
        deployment=AZURE_DEPLOYMENT_NAME,
        api_version=AZURE_API_VERSION,
        subscription_key=os.environ.get("OPENAI_API_KEY"),
        proxy=os.environ.get("HTTP_PROXY")
    )

    try:
        test_response = llm.invoke("Hello, please reply with OK.")
        print("✅ LLM Connected:", test_response)
    except Exception as e:
        print("⚠️ LLM smoke test failed:", repr(e))

    return llm


def convert_urls_to_markdown(text):
    url_pattern = r'(https?://[^\s]+)'
    return re.sub(url_pattern, r'[\1](\1)', text)


def setup_rag_chain(llm, vector_database):
    """Corrected RAG chain — chunks go into USER context, not system."""

    system_prompt = (
        "You are a highly accurate enterprise document Q&A bot."
        "Use the provided context to answer the user's question as best as possible."
        "If there are any URLs in the context, provide them as clickable Markdown links."
        "If the context does not fully answer the question, provide a cautious answer based on the context."
        "Avoid hallucination and clearly indicate uncertainty if needed."
    )

    K = 10
    TOP_N = 3

    def rag_call(inputs):
        query = inputs.get("question", "").strip()
        if not query:
            return "I cannot find the answer in the provided documents."

        # --- Retrieve ---
        try:
            docs = vector_database.similarity_search(query, k=K)
        except Exception as e:
            print("Retriever error:", repr(e))
            docs = []

        # --- Clean / dedupe ---
        cleaned = []
        seen = set()
        for doc in docs:
            text = doc.page_content.strip().lstrip(" .\n\t:,-—")
            if text and text not in seen:
                cleaned.append(text)
                seen.add(text)

        if not cleaned:
            return "I cannot find the answer in the provided documents."

        # --- Build context block sent in USER message ---
        context_text = "\n\n".join([doc.page_content for doc in docs[:6]])
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"}
                ]               

        # Debug view
        print("\n================== CONTEXT SENT TO MODEL ==================")
        print(context_text[:1500])
        print("\n===========================================================")

        # --- Correct message structure ---
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question:\n{query}"
                )
            }
        ]

        # --- Call LLM ---
        try:
            response = llm.invoke_chat(messages)
            response = convert_urls_to_markdown(response)
            print("LLM RAW RESPONSE:", response)
        except Exception as e:
            print("LLM error:", repr(e))
            return "I cannot find the answer in the provided documents."

        return response

    return RunnableLambda(rag_call)


if __name__ == "__main__":

    # Populate vector DB if empty
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        print("ChromaDB empty. Populating...")
        docs = load_documents(DATA_PATH)
        chunks = semantic_chunk_free_local(docs, threshold_percentile=90)
        _, embed = get_vector_database()
        store_chunks_in_chromadb(chunks, embed, db_path=CHROMA_DB_PATH)

    vectordb, _ = get_vector_database(CHROMA_DB_PATH)
    llm = initialize_llm()
    rag_chain = setup_rag_chain(llm, vectordb)

    print("\n🎉 RAG ready!\n")

    while True:
        q = input("\nQuery: ").strip()
        if q.lower() == "exit":
            break
        result = rag_chain.invoke({"question": q})
        print("\n🤖 ANSWER:\n", result)
