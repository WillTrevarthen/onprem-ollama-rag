import os
import sys
import re

# --- 1. Imports for LangChain v1.1.0+ ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Vector Store & Models
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Core Logic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

try:
    # In LangChain v1.x, these moved to langchain_classic
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError:
    # Fallback for older environments (just in case)
    try:
        from langchain.retrievers import EnsembleRetriever
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        print("CRITICAL: Could not find 'EnsembleRetriever'.")
        print("Please ensure 'langchain-classic' is installed: pip install langchain-classic")
        sys.exit(1)

# Community Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# Configuration
MODEL_NAME = 'llama3.1'
EMBED_MODEL = 'mxbai-embed-large'
VECTOR_STORE_PATH = 'chroma_db'
COLLECTION_NAME = 'test_pdfs'

class RAGChatBot:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
        print(f"Initializing RAG Pipeline with {MODEL_NAME}...")
        
        # 1. Initialize Embedding Model
        self.embedding_function = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url='http://localhost:11434'
        )

        # 2. Initialize LLM
        self.llm = ChatOllama(
            model=MODEL_NAME,
            temperature=0.2,
            base_url='http://localhost:11434'
        )

        # 3. Initialize Vector Store
        self.vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=self.embedding_function,
            collection_name=COLLECTION_NAME
        )
        
        # 4. Initialize Reranker 
        print("Loading Reranker Model (FlashRank)...")
        try:
            self.reranker = FlashrankRerank()
        except Exception as e:
            print(f"Warning: Flashrank failed to load. Error: {e}")
            self.reranker = None

    def get_indexed_files(self):
        """Returns a list of filenames currently in the database."""
        data = self.vector_store.get()
        if data and data['metadatas']:
            return {m.get('source_file') for m in data['metadatas'] if m}
        return set()

    def delete_pdf(self, filename):
        """Removes a file from the ChromaDB."""
        try:
            print(f"Attempting to delete {filename}...")
            # Chroma internal collection delete by metadata
            self.vector_store._collection.delete(where={"source_file": filename})
            print("Deleted from DB.")
            return True
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return False

    def load_and_index_pdfs(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            return

        local_files = {f for f in os.listdir(self.folder_path) if f.endswith('.pdf')}
        
        # Check existing
        existing_data = self.vector_store.get()
        indexed_files = set()
        if existing_data and existing_data['metadatas']:
            indexed_files = {m.get('source_file') for m in existing_data['metadatas'] if m}

        new_files = list(local_files - indexed_files)

        if not new_files:
            print("No new files to index.")
            return

        print(f"Found {len(new_files)} new files to process.")
        
        all_chunks = []
        
        for filename in new_files:
            file_path = os.path.join(self.folder_path, filename)
            print(f"Loading: {filename}")
            
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                # --- METADATA INJECTION LOGIC ---
                current_chapter = "Unknown Chapter"
                
                for doc in docs:
                    # 1. Extract Page Number (PyPDFLoader indexes start at 0)
                    page_num = doc.metadata.get('page', 0) + 1
                    
                    # 2. simple Chapter Detection (Regex)
                    # Looks for lines starting with "Chapter X" or "1.0 Introduction"
                    # You can customize this regex based on your specific PDF style
                    content = doc.page_content
                    chapter_match = re.search(
                        r'^(Law \d+|SECTION [A-Z0-9]+|Rule \d+|[A-Z\s]{5,})', 
                        content, 
                        re.MULTILINE
                    )
                    if chapter_match:
                        current_chapter = chapter_match.group(0)

                    # 3. Inject Metadata into the Text
                    # We prepend this info so the embedding model reads it first.
                    # This makes the "Where" just as important as the "What".
                    contextualized_content = (
                        f"Source: {filename}\n"
                        f"Page: {page_num}\n"
                        f"Chapter: {current_chapter}\n"
                        f"----------------\n"
                        f"{content}"
                    )
                    
                    # Update the doc object
                    doc.page_content = contextualized_content
                    doc.metadata['source_file'] = filename
                    doc.metadata['page_number'] = page_num
                    doc.metadata['chapter'] = current_chapter

                # 4. Split (Now splitting the Contextualized text)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500, # Increased slightly to account for added metadata
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if all_chunks:
            print(f"Embedding and indexing {len(all_chunks)} chunks with metadata...")
            self.vector_store.add_documents(all_chunks)
            print("Indexing complete.")

        if all_chunks:
            print(f"Indexing {len(all_chunks)} chunks...")
            self.vector_store.add_documents(all_chunks)
            print("Indexing complete.")

    def _build_hybrid_retriever(self, top_k):
            data = self.vector_store.get()
            if not data or not data['documents']:
                return None
                
            docs = [
                Document(page_content=txt, metadata=md or {}) 
                for txt, md in zip(data['documents'], data['metadatas'])
            ]

            fetch_k = 50 

            # Keyword Search (Fetch 50)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = fetch_k

            # Semantic Search (Fetch 50)
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": fetch_k})

            # 3. Combine (We now have up to 100 candidates)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            
            # Rerank (Smart Filter)
            if self.reranker:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker, 
                    base_retriever=ensemble_retriever
                )
                return compression_retriever
            else:
                return ensemble_retriever

    def query(self, question, top_k=5):
        # Generate Hypothetical Answer
        print("Generating HyDE document...")
        hyde_prompt = f"""Given the question '{question}', write a paragraph that would answer it. 
        Do not say you don't know. Just make up a plausible answer using technical keywords."""
        
        hypothetical_answer = self.llm.invoke(hyde_prompt).content
        
        # Search using the Hypothetical Answer instead of the Question
        combined_query = f"{question} {hypothetical_answer}"
        
        retriever = self._build_hybrid_retriever(top_k=top_k)
        
        if not retriever:
            return "The database is empty. Please add PDFs first."

        template = """You are an expert research assistant. Use the provided context to answer the question.
                
                Rules:
                1. Answer ONLY using the provided context. If the answer is not there, say "I don't know."
                2. Think step-by-step before answering.
                3. Cite the source file and page number for every fact you state.
                
                Format your answer like this:
                [Answer]
                (Source: filename.pdf, Page: X)

                Context:
                {context}
                
                Question: {question}
                """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(question)

if __name__ == "__main__":
    # Standard terminal usage test
    bot = RAGChatBot(folder_path="pdfs")
    bot.load_and_index_pdfs()
    
    print("\nSystem Ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuery: ")
        if query.lower() == 'exit':
            break
        print("\n" + bot.query(query) + "\n")