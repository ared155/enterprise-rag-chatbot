import streamlit as st
import os

# --- Import your existing functions ---
from load_chunk import load_documents, semantic_chunk_free_local, DATA_PATH, LOCAL_MODEL_PATH
from vectordb_manager import get_vector_database, store_chunks_in_chromadb, CHROMA_DB_PATH
from rag_pipeline_1 import initialize_llm, setup_rag_chain

# --- Streamlit page config ---
st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")
st.title("SQL Chat-BOT")

# --- Setup function (cached to avoid reloading every run) ---
@st.cache_resource
def setup_database_and_rag():
    """Ensure DB is ready and initialize RAG chain."""
    # Populate DB if missing
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        with st.spinner("Populating vector database..."):
            docs = load_documents(DATA_PATH)
            if not docs:
                st.error("No documents found. Check your DATA_PATH!")
                st.stop()
            chunks = semantic_chunk_free_local(docs, threshold_percentile=90)
            vectordb, embed_func = get_vector_database(db_path=CHROMA_DB_PATH)
            store_chunks_in_chromadb(chunks, embed_func, db_path=CHROMA_DB_PATH)
            st.success("Vector database populated.")

    # Load DB
    vectordb, _ = get_vector_database(db_path=CHROMA_DB_PATH)
    llm = initialize_llm()
    rag_chain = setup_rag_chain(llm, vectordb)
    return rag_chain

# --- Initialize RAG chain ---
rag_chain = setup_database_and_rag()

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"question": prompt})
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
