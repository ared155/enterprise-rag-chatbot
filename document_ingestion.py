# load_and_chunk.py

# load_and_chunk.py

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os
import re # Import the regex library for cleaning

DATA_PATH = "data"
LOCAL_MODEL_PATH = "./all-MiniLM-L6-v2/" 

# ---------------------------
# Custom UTF-8 HTML Loader
# ---------------------------

def clean_html_text(text: str) -> str:
    """
    Cleans up excessive whitespace, newlines, and tabs from text extracted from HTML.
    """
    # Replace multiple newlines with a single space or period-space for better splitting
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

class UTF8HTMLLoader:
    """Load an HTML file as a LangChain Document with UTF-8 encoding and clean it."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            html = f.read()
        raw_text = BeautifulSoup(html, "html.parser").get_text()
        cleaned_text = clean_html_text(raw_text)
        return [Document(page_content=cleaned_text, metadata={"source": self.file_path})]


def load_documents(data_path=DATA_PATH):
    """Load all HTML files in the given directory."""
    documents = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(".html"):
                loader = UTF8HTMLLoader(os.path.join(root, file))
                documents.extend(loader.load())
    return documents


def semantic_chunk_free_local(documents: list[Document], threshold_percentile=90):
    """
    Splits documents into semantically coherent chunks using a locally saved, free model.
    """
    
    print(f"Attempting to load model from local path: {os.path.abspath(LOCAL_MODEL_PATH)}")
    
    if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, 'modules.json')):
        print(f"Error: modules.json not found in the specified directory: {LOCAL_MODEL_PATH}")
        return []

    model = SentenceTransformer(LOCAL_MODEL_PATH)
    
    chunks = []
    for doc in documents:
        text = doc.page_content # This text should now be much cleaner

        # 1. Split text into potential "sentences" or small segments
        # We increase the chunk_size here because we want *longer* initial segments
        # that make sense for generating meaningful embeddings.
        sentence_splitter = RecursiveCharacterTextSplitter(
            separators=[". ", "? ", "! ", "\n\n", "\n", " "],
            chunk_size=500, # Increased size for better initial segmentation
            chunk_overlap=50,
            length_function=len
        )
        # Filter out any segments that are just empty strings or single spaces
        sentences = [s.strip() for s in sentence_splitter.split_text(text) if s.strip()]
        
        if len(sentences) <= 1:
            chunks.append(Document(page_content=text, metadata=doc.metadata))
            continue

        # 2. Generate embeddings for sentences locally
        embeddings = model.encode(sentences)
        
        # 3. Calculate cosine similarity between adjacent sentences
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] # Access the single value
            for i in range(len(embeddings) - 1)
        ]
        
        # 4. Detect breakpoints using a percentile threshold
        breakpoint_threshold = np.percentile(similarities, threshold_percentile)
        breakpoints = [
            i + 1 for i, similarity in enumerate(similarities)
            if similarity < breakpoint_threshold
        ]
        
        # 5. Combine sentences into final chunks
        start_index = 0
        for i, end_index in enumerate(breakpoints):
            chunk_content = " ".join(sentences[start_index:end_index]).strip()
            if chunk_content:
                chunks.append(Document(page_content=chunk_content, metadata={**doc.metadata, "chunk_id": i + 1}))
            start_index = end_index
            
        # Add the final chunk
        final_chunk_content = " ".join(sentences[start_index:]).strip()
        if final_chunk_content:
             chunks.append(Document(page_content=final_chunk_content, metadata={**doc.metadata, "chunk_id": len(breakpoints) + 1}))
    
    print(f"✅ Created {len(chunks)} free semantic chunks locally.")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = semantic_chunk_free_local(docs, threshold_percentile=90)

        # Display preview
        if chunks:
            print(f"\nTotal documents loaded: {len(docs)}")
            print(f"Total chunks created: {len(chunks)}")
            # print("\n🔍 Sample chunk (first 500 chars):")
            # print(chunks[0].page_content[:500], "...")
        else:
            print("No chunks were generated.")
    else:
        print("No documents were loaded from the data directory.")

