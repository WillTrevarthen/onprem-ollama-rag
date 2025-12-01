import streamlit as st
import os
import shutil

# Import the class from your other file
# Make sure rag_backend.py is in the same folder as app.py
from rag_backend import RAGChatBot

# Configuration
PDF_FOLDER = "test_pdfs"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize Session State for the Bot
# This ensures we don't reload the models every time you click a button
if "bot" not in st.session_state:
    with st.spinner("Initializing AI Models..."):
        st.session_state.bot = RAGChatBot(folder_path=PDF_FOLDER)
        # Run an initial check for files
        st.session_state.bot.load_and_index_pdfs()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar (Upload & Delete) ---
with st.sidebar:
    st.header("📂 Document Manager")
    
    # 1. Upload
    uploaded_files = st.file_uploader("Upload new PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Uploads"):
            # Ensure folder exists
            if not os.path.exists(PDF_FOLDER):
                os.makedirs(PDF_FOLDER)
                
            for file in uploaded_files:
                file_path = os.path.join(PDF_FOLDER, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            # Trigger Indexing in Backend
            with st.spinner("Indexing documents..."):
                st.session_state.bot.load_and_index_pdfs()
            st.success("Files uploaded and indexed!")
            st.rerun()

    st.divider()

    # 2. Delete
    st.subheader("Indexed Documents")
    
    # Get list of files currently in the Vector DB
    current_files = list(st.session_state.bot.get_indexed_files())
    
    if current_files:
        for filename in current_files:
            col1, col2 = st.columns([0.8, 0.2])
            col1.text(filename)
            # Unique key required for every button
            if col2.button("🗑️", key=f"del_{filename}"):
                # 1. Remove from Vector DB
                st.session_state.bot.delete_pdf(filename)
                
                # 2. Remove from Disk (Optional, but keeps folder clean)
                file_path = os.path.join(PDF_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                st.toast(f"Deleted {filename}")
                st.rerun()
    else:
        st.info("No documents found in database.")

# --- Main Chat Area ---
st.title("🤖 RAG Assistant")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
if prompt := st.chat_input("Ask about your documents..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call the backend query method
                response = st.session_state.bot.query(prompt)
                st.markdown(response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")