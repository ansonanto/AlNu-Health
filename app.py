import os
import time
import logging
import streamlit as st
import openai

# Import configuration
from config import OPENAI_API_KEY, GEMINI_API_KEY, VECTOR_STORE_PATH

# Import modules
from document_processor import PaperManager, process_documents
from vector_db import initialize_vector_db, create_vector_db, check_db_status
from query_processor import query_documents, generate_accuracy_percentage
from pubmed_downloader import pubmed_downloader_ui
from prompt_evaluator import prompt_evaluator_ui
from keyword_availability import keyword_availability_ui
from gemini_services import gemini_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = OPENAI_API_KEY

# Set page configuration
st.set_page_config(
    page_title="AlNu Health - RAG Document Search System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'db' not in st.session_state:
    st.session_state.db = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'db_status' not in st.session_state:
    st.session_state.db_status = "Not initialized"
if 'new_documents' not in st.session_state:
    st.session_state.new_documents = []
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = None

# Initialize additional session state variables
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'accuracy_percentage' not in st.session_state:
    st.session_state.accuracy_percentage = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = None
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = True  # Default to chat mode enabled
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Document Management"

def main():
    """Main application UI"""
    # Check for existing vector store at startup
    if 'db_initialized' not in st.session_state:
        # Initialize vector store
        db = initialize_vector_db()
        
        # Check if the database has documents
        try:
            if db is not None:
                st.session_state.db = db
                st.session_state.db_initialized = True
                st.session_state.db_status = "Vector store loaded successfully"
            else:
                st.session_state.db_initialized = False
                st.session_state.db_status = "Vector store not initialized"
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            st.session_state.db_initialized = False
            st.session_state.db_status = f"Error: {str(e)}"
        
        # If vector store initialization failed, set appropriate status
        if 'db_status' in st.session_state and st.session_state.db_status and st.session_state.db_status.startswith("Error"):
            logger.warning("Vector store initialization failed, some features will be limited")
            st.warning("Vector store initialization failed. Some features may be limited.")
        
        # Mark as initialized so we don't check again
        st.session_state.db_initialized = True
    
    # Display header
    st.title("AlnuHealth - Medical Research RAG System")
    
    # Create tabs for different functionalities
    tabs = ["Document Management", "Search & Query", "PubMed Downloader", "Prompt Evaluator", "Keyword Availability"]
    st.session_state.current_tab = st.radio("Select Functionality:", tabs, horizontal=True)
    
    # Check if ChromaDB is available and show warning if not
    if 'db_status' in st.session_state and st.session_state.db_status and st.session_state.db_status.startswith("Error"):
        st.warning(f"⚠️ Vector database is not available: {st.session_state.db_status}. Some features will be limited.")
        st.info("You can still use the PubMed Downloader and Prompt Evaluator features.")
    
    # Display the selected functionality
    if st.session_state.current_tab == "Document Management":
        paper_manager = PaperManager()
        paper_manager.show_ui()
    elif st.session_state.current_tab == "Search & Query":
        query_documents()
    elif st.session_state.current_tab == "PubMed Downloader":
        pubmed_downloader_ui()
    elif st.session_state.current_tab == "Prompt Evaluator":
        prompt_evaluator_ui()
    elif st.session_state.current_tab == "Keyword Availability":
        keyword_availability_ui()

if __name__ == "__main__":
    main()
