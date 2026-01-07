# vectordb_manager.py

import os
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# Import functions from our chunking utility file
from document_ingestion import load_documents, semantic_chunk_free_local, LOCAL_MODEL_PATH 

# --- Configuration ---
CHROMA_DB_PATH = "./chroma" 
# ---------------------

# ---------------------------
# Custom Embedding Wrapper 
# ---------------------------
class LocalSentenceTransformerEmbeddings:
    """
    A custom wrapper to make the local SentenceTransformer model work 
    with LangChain's Chroma vector store integration.
    """
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts).tolist()
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(query).tolist()
        return embedding

def get_vector_database(db_path=CHROMA_DB_PATH, model_path=LOCAL_MODEL_PATH):
    """
    Initializes and returns a connection to the ChromaDB instance.
    If the DB exists, it loads it. Otherwise, it prepares for creation.
    """
    embedding_function = LocalSentenceTransformerEmbeddings(model_path)
    
    # Check if DB already exists
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Loading existing ChromaDB from: {db_path}")
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            collection_name="html_chunks_collection"
        )
    else:
        print(f"ChromaDB not found. It will be created when data is added.")
        vectordb = None

    return vectordb, embedding_function


def store_chunks_in_chromadb(chunks: list[Document], embedding_function, db_path=CHROMA_DB_PATH):
    """
    Generates embeddings for chunks and stores them in a persistent ChromaDB instance.
    """
    print(f"\nAttempting to store {len(chunks)} chunks in ChromaDB at: {os.path.abspath(db_path)}")
        
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=db_path, 
        collection_name="html_chunks_collection"
    )
    
    vectordb.persist()
    print(f"✅ Successfully stored embeddings in ChromaDB.")
    return vectordb


if __name__ == "__main__":
    # Main execution flow: Load -> Chunk -> Store in DB

    # 1. Load the documents using function from the other file
    docs = load_documents()
    
    if docs:
        # 2. Chunk the documents using function from the other file
        chunks = semantic_chunk_free_local(docs, threshold_percentile=90)

        # 3. Store the chunks in ChromaDB
        if chunks:
            # We get the embedding function we will use
            _, embed_func = get_vector_database() 
            
            # Store data
            vector_database = store_chunks_in_chromadb(chunks, embed_func)
            print("\nDatabase is ready for retrieval!")
            
            # Example of how you can immediately test the retrieval
            query = "What does migration do ?"
            print(f"\nTesting retrieval for query: '{query}'")
            results = vector_database.similarity_search(query, k=3)
            print("Retrieved result (top 3 chunk):")
            print(results[0].page_content)

        else:
            print("No chunks were generated, skipping database storage.")
    else:
        print("No documents were loaded from the data directory. Check your DATA_PATH in chunking_utils.py.")
