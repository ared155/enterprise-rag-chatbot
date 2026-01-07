# enterprise-rag-chatbot
Enterprise-grade Retrieval-Augmented Generation (RAG) system using local embeddings, ChromaDB, FAISS retrieval, and Azure-based LLM inference with Streamlit and CLI interfaces.

## Architecture

```mermaid
flowchart LR
    A[HTML Documents<br/>data/*.html] --> B[UTF-8 Loader<br/>BeautifulSoup]
    B --> C[Text Cleaning<br/>Regex]
    C --> D[Semantic Chunking]
    
    D --> E[Local Embeddings<br/>Sentence Transformers]
    E --> F[ChromaDB<br/>Persistent Store]

    F --> G[FAISS Index<br/>In-Memory]

    H[User Query] --> I[Query Embedding]
    I --> G

    G --> J[Top-K Context]
    J --> K[RAG Prompt Builder]
    K --> L[Azure Farm GPT]
    L --> M[Answer]
    M --> N[Streamlit UI]
