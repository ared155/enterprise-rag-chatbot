# enterprise-rag-chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed for document-based question answering using local embeddings, persistent vector storage, FAISS retrieval, and Azure-based LLM inference.

This project demonstrates a **production-style GenAI architecture**, not a demo or tutorial.

---

## 🚀 Key Highlights

- 🔹 Free **local sentence-transformer embeddings** (no OpenAI embedding cost)
- 🔹 **Semantic chunking** for higher retrieval quality
- 🔹 **ChromaDB** for persistent vector storage
- 🔹 **FAISS** for fast in-memory similarity search
- 🔹 Azure Farm GPT integration via a **custom LLM client**
- 🔹 Dual interface: **Streamlit UI + CLI**
- 🔹 Correct RAG prompting (context injected into user message)

---

## 🧠 System Architecture

```mermaid
flowchart LR
    A[HTML Documents<br/>data/*.html] --> B[UTF-8 Loader<br/>BeautifulSoup]
    B --> C[Text Cleaning<br/>Regex]
    C --> D[Semantic Chunking]
    
    D --> E[Local Embeddings<br/>Sentence Transformers]
    E --> F[ChromaDB<br/>Persistent Store]

    F --> G[FAISS Index<br/>In-Memory Retrieval]

    H[User Query] --> I[Query Embedding<br/>Local Model]
    I --> G

    G --> J[Top-K Context Chunks]
    J --> K[RAG Prompt Builder]
    K --> L[Azure Farm GPT]
    L --> M[Final Answer]
    M --> N[Streamlit Chat UI]
