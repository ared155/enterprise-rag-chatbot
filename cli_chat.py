# query.py - The interactive interface to your RAG system using FAISS for search

import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
# Import utility functions (ensure these files are correct as per previous steps)
from document_ingestion import load_documents, semantic_chunk_free_local, DATA_PATH, LOCAL_MODEL_PATH 
from vector_store import get_vector_database, store_chunks_in_chromadb, CHROMA_DB_PATH, LocalSentenceTransformerEmbeddings
from rag_chain import initialize_llm, setup_rag_chain_with_faiss_retriever # Import the specific FAISS setup function


if __name__ == "__main__":
    
    # --- Step 1: Ensure DB is Populated in Chroma (Storage layer) ---
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        print("🔍 No existing Chroma DB found → Creating it now...")
        docs = load_documents(DATA_PATH)
        if not docs:
            print("No source documents found. Cannot populate the database. Exiting.")
            exit()
        chunks = semantic_chunk_free_local(docs, model_path=LOCAL_MODEL_PATH)
        if chunks:
            _, embed_func = get_vector_database(db_path=CHROMA_DB_PATH, model_path=LOCAL_MODEL_PATH)
            store_chunks_in_chromadb(chunks, embed_func, db_path=CHROMA_DB_PATH)
            print("✅ Database created and populated in ChromaDB.")
        else:
            print("Could not chunk documents. Exiting.")
            exit()
    else:
        print("✅ Found existing Chroma DB.")


    # --- Step 2: Load Data FROM Chroma INTO FAISS (Retrieval layer) ---
    # This satisfies the requirement that the search operation uses FAISS
    print("🔄 Loading data from ChromaDB into an in-memory FAISS index...")
    
    # We load the data using the embedding function wrapper
    embedding_func = LocalSentenceTransformerEmbeddings(LOCAL_MODEL_PATH)
    
    # Load Chroma (reads the data off the disk into memory)
    chroma_db = get_vector_database(db_path=CHROMA_DB_PATH, model_path=LOCAL_MODEL_PATH)[0]
    
    # Extract all data from Chroma
    all_chroma_data = chroma_db.get(include=["embeddings", "documents", "metadatas"])

    # Create a new FAISS index using this extracted data
    # The 'embedding_function' argument tells FAISS which function to use when a query is invoked
    faiss_vector_store = FAISS.from_embeddings(
        text_embeddings=zip(all_chroma_data["documents"], all_chroma_data["embeddings"]),
        embedding=embedding_func, # <-- CHANGED to 'embedding='
        metadatas=all_chroma_data["metadatas"],
    )
    
    print("✅ FAISS index created in memory, ready for search.")


    # --- Step 3: Initialize Components and RAG Chain ---
    llm = initialize_llm()
    # Use the function tailored for a FAISS input
    rag_chain = setup_rag_chain_with_faiss_retriever(llm, faiss_vector_store)


    # --- Step 4: Start Interactive Chat Loop ---
    print("\n🎉 Ready! Ask your questions:")
    while True:
        q = input("\n> ")
        if q == "exit": 
            print("Shutting down query interface.")
            break
        if not q.strip(): continue

        print(f"🤖 AI Response:")
        # The invoke call triggers the FAISS retriever, which embeds the query and searches
        print(rag_chain.invoke(q))
