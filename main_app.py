import os
import time
import logging
import streamlit as st
import openai

# Import configuration
from config import OPENAI_API_KEY, CHROMA_PATH

# Import utility functions
from utils import reset_chroma, verify_chroma_persistence

# Import modules
from document_processor import PaperManager, process_documents
from vector_db import initialize_chroma, create_vector_db, check_db_status
from query_processor import query_documents, generate_accuracy_percentage
from pubmed_downloader import pubmed_downloader_ui
from prompt_evaluator import prompt_evaluator_ui

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
if 'chroma_instance' not in st.session_state:
    st.session_state.chroma_instance = None

# Initialize additional session state variables for search results and UI state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'accuracy_percentage' not in st.session_state:
    st.session_state.accuracy_percentage = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Document Management"
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False

def main():
    """Main application UI"""
    # Display header
    st.title("AlNu Health - Medical Research RAG System")
    
    # Create tabs for different functionalities
    tabs = ["Document Management", "Search & Query", "PubMed Downloader", "Prompt Evaluator"]
    st.session_state.current_tab = st.radio("Select Functionality:", tabs, horizontal=True)
    
    # Display the selected tab
    if st.session_state.current_tab == "Document Management":
        document_management_ui()
    elif st.session_state.current_tab == "Search & Query":
        search_query_ui()
    elif st.session_state.current_tab == "PubMed Downloader":
        pubmed_downloader_ui()
    elif st.session_state.current_tab == "Prompt Evaluator":
        prompt_evaluator_ui()
    
    # Display footer
    st.markdown("---")
    st.markdown("AlNu Health - Medical Research RAG System © 2025")

def document_management_ui():
    """UI for document management tab"""
    st.header("Document Management")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Document processing section
        st.subheader("Document Processing")
        
        # Check database status
        check_db_status()
        
        # Display database status
        st.info(f"Vector Database Status: {st.session_state.db_status}")
        
        # Process documents button
        if st.button("Process Documents"):
            # Process documents
            documents, new_docs = process_documents()
            
            if documents:
                st.session_state.documents = documents
                st.session_state.new_documents = new_docs
                
                # Create vector database
                db = create_vector_db(documents)
                if db:
                    st.session_state.db = db
                    st.success(f"Successfully processed {len(documents)} documents")
                else:
                    st.error("Failed to create vector database")
            else:
                st.warning("No documents to process")
        
        # Reset database button
        if st.button("Reset Database"):
            if reset_chroma():
                # Clear session state
                st.session_state.db = None
                st.session_state.processed_docs = False
                st.session_state.chroma_instance = None
                st.success("Vector database reset successfully")
            else:
                st.error("Failed to reset vector database")
        
        # Display document info
        if st.session_state.processed_docs:
            st.subheader("Processed Documents")
            
            # Initialize PaperManager
            paper_manager = PaperManager()
            
            # Get paper info
            num_papers, paper_titles = paper_manager.get_paper_info()
            
            # Display paper info
            st.info(f"Number of papers in database: {num_papers}")
            
            # Display paper titles
            if paper_titles:
                st.write("Paper titles:")
                for title in paper_titles:
                    st.write(f"- {title}")
            else:
                st.warning("No paper titles found")
    
    with col2:
        # Document upload section
        st.subheader("Document Upload")
        
        # Display instructions
        st.markdown("""
        ### Instructions
        1. Upload PDF files to the 'results' directory
        2. Click 'Process Documents' to extract text and create embeddings
        3. Use the 'Search & Query' tab to ask questions about the documents
        """)
        
        # Display last processed time
        if st.session_state.last_processed_time:
            last_processed = time.strftime(
                "%Y-%m-%d %H:%M:%S", 
                time.localtime(st.session_state.last_processed_time)
            )
            st.info(f"Last processed: {last_processed}")

def search_query_ui():
    """UI for search and query tab"""
    st.header("Search & Query")
    
    # Check if documents have been processed
    if not st.session_state.processed_docs or st.session_state.db is None:
        st.warning("Please process documents first in the Document Management tab")
        return
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        st.subheader("Ask a Question")
        query = st.text_area("Enter your question:", height=100)
        
        # Query button
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    # Get conversation history
                    conversation_history = st.session_state.query_history[-5:] if st.session_state.query_history else []
                    
                    # Query documents
                    results = query_documents(query, st.session_state.db, conversation_history)
                    
                    # Store results
                    st.session_state.search_results = results
                    
                    # Generate accuracy percentage
                    st.session_state.accuracy_percentage = generate_accuracy_percentage()
                    
                    # Add to query history
                    st.session_state.query_history.append({
                        "user": query,
                        "assistant": results["response"]
                    })
            else:
                st.warning("Please enter a question")
        
        # Display results
        if st.session_state.search_results:
            st.subheader("Answer")
            
            # Display accuracy percentage
            if st.session_state.accuracy_percentage:
                st.info(f"Response Confidence: {st.session_state.accuracy_percentage}%")
            
            # Display response
            st.markdown(st.session_state.search_results["response"])
            
            # Display processing time
            processing_time = st.session_state.search_results.get("processing_time", 0)
            st.caption(f"Processing time: {processing_time:.2f} seconds")
            
            # Display sources
            if st.session_state.search_results.get("sources"):
                st.subheader("Sources")
                sources = st.session_state.search_results["sources"]
                for i, source in enumerate(sources):
                    st.write(f"{i+1}. {source}")
    
    with col2:
        # Query history
        st.subheader("Query History")
        
        if st.session_state.query_history:
            for i, exchange in enumerate(st.session_state.query_history):
                with st.expander(f"Q{i+1}: {exchange['user'][:30]}..."):
                    st.write("**Question:**")
                    st.write(exchange["user"])
                    st.write("**Answer:**")
                    st.write(exchange["assistant"])
        else:
            st.info("No queries yet")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.success("Query history cleared")

if __name__ == "__main__":
    main()
