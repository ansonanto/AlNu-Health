import os
import time
import logging
import sqlite3
import streamlit as st
import openai

# Import configuration
from config import OPENAI_API_KEY, CHROMA_PATH

# Import utility functions
from utils import reset_chroma, verify_chroma_persistence

# Import modules
from document_processor import PaperManager, process_documents
from vector_db import initialize_vector_db, create_vector_db, check_db_status
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
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False

def main():
    """Main application UI"""
    # Check for existing ChromaDB data at startup
    if 'db_initialized' not in st.session_state:
        # Initialize vector database (will use FAISS if ChromaDB isn't available)
        db = initialize_vector_db()
        
        # Check if the database has documents
        try:
            # Try to get the collection count if it's ChromaDB
            if db and hasattr(db, '_collection'):
                count = db._collection.count()
                logger.info(f"Found {count} documents in vector database at startup")
                
                if count > 0:
                    # Database exists and has documents
                    st.session_state.db = db
                    st.session_state.processed_docs = True
                    if 'db_status' not in st.session_state or not st.session_state.db_status:
                        st.session_state.db_status = "Healthy (Loaded from disk)"
                    logger.info("Successfully loaded existing vector database")
        except Exception as e:
            logger.error(f"Error checking vector database at startup: {str(e)}")
            
        # If initialization failed, set appropriate status
        if 'db_status' in st.session_state and st.session_state.db_status and st.session_state.db_status.startswith("Error"):
            logger.warning("Vector database initialization failed, some features will be limited")
        
        # Mark as initialized so we don't check again
        st.session_state.db_initialized = True
    
    # Display header
    st.title("AlNu Health - Medical Research RAG System")
    
    # Create tabs for different functionalities
    tabs = ["Document Management", "Search & Query", "PubMed Downloader", "Prompt Evaluator"]
    st.session_state.current_tab = st.radio("Select Functionality:", tabs, horizontal=True)
    
    # Check if vector database is available and show appropriate messages
    if 'db_status' in st.session_state and st.session_state.db_status:
        if st.session_state.db_status.startswith("Error"):
            st.warning(f"⚠️ Vector database is not available: {st.session_state.db_status}")
            st.info("You can still use the PubMed Downloader and Prompt Evaluator features which don't require a vector database.")
        elif "FAISS" in st.session_state.db_status:
            st.success("Using FAISS as the vector database (fallback mode)")
            st.info("ChromaDB couldn't be used due to SQLite version compatibility, but FAISS is working as a fallback.")
            st.info("All features should work normally with FAISS.")
        elif "ChromaDB" in st.session_state.db_status:
            st.success("Using ChromaDB as the vector database")
        else:
            st.info(f"Vector database status: {st.session_state.db_status}")
    
    # Display the selected tab
    if st.session_state.current_tab == "Document Management":
        if 'db_status' in st.session_state and st.session_state.db_status and st.session_state.db_status.startswith("Error"):
            st.error("Document Management requires the vector database, which is not available in this environment.")
        else:
            document_management_ui()
    elif st.session_state.current_tab == "Search & Query":
        if 'db_status' in st.session_state and st.session_state.db_status and st.session_state.db_status.startswith("Error"):
            st.error("Search & Query requires the vector database, which is not available in this environment.")
        else:
            search_query_ui()
    elif st.session_state.current_tab == "PubMed Downloader":
        pubmed_downloader_ui()
    elif st.session_state.current_tab == "Prompt Evaluator":
        prompt_evaluator_ui()
    
    # Display footer
    st.markdown("---")
    st.markdown("AlNu Health - Medical Research RAG System 2025")

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
        if st.session_state.db_status and st.session_state.db_status.startswith("Error"):
            st.error(f"Vector Database Status: {st.session_state.db_status}")
            st.info("This is likely due to the SQLite version on the deployment environment being older than what ChromaDB requires.")
            st.info("You can still use the PubMed Downloader and Prompt Evaluator features.")
        else:
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
    
    # Chat mode toggle
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col1:
        st.subheader("Medical Research Assistant")
    with chat_col2:
        st.session_state.chat_mode = st.toggle("Continuous Chat", value=st.session_state.chat_mode)
        if st.session_state.chat_mode:
            st.caption("Chat history is maintained")
        else:
            st.caption("Each query is independent")
    
    # Check if documents have been processed
    if not st.session_state.processed_docs:
        st.warning("Please process documents first in the Document Management tab.")
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
                    # Only use conversation history if chat mode is enabled
                    conversation_history = []
                    if st.session_state.chat_mode:
                        conversation_history = st.session_state.query_history if st.session_state.query_history else []
                        
                        # Check if this might be a follow-up question
                        is_follow_up = False
                        follow_up_indicators = ["it", "this", "that", "they", "them", "those", "these", "their", "what about", "how about", "what else", "tell me more", "and", "also", "too", "as well"]
                        
                        # Simple heuristic to detect follow-up questions
                        query_lower = query.lower()
                        if conversation_history:
                            # Check for follow-up indicators at the start of the query
                            for indicator in follow_up_indicators:
                                if query_lower.startswith(indicator + " ") or " " + indicator + " " in query_lower:
                                    is_follow_up = True
                                    break
                        
                        if is_follow_up:
                            st.info("This appears to be a follow-up question. Using conversation history for context.")
                    else:
                        # If chat mode is disabled, clear the conversation memory
                        if 'conversation_memory' in st.session_state and st.session_state.conversation_memory is not None:
                            st.session_state.conversation_memory = None
                            logger.info("Chat mode disabled. Cleared conversation memory.")
                    
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
                chunks = st.session_state.search_results.get("chunks", [])
                
                # Create an expander for each source to show more details
                for i, source in enumerate(sources):
                    with st.expander(f"Source {i+1}: {source}"):
                        if i < len(chunks):
                            st.markdown("**Excerpt:**")
                            st.text(chunks[i][:500] + "..." if len(chunks[i]) > 500 else chunks[i])
    
    with col2:
        # Chat history
        st.subheader("Conversation History")
        
        if st.session_state.query_history:
            # Add controls for conversation history
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Conversation"):
                    st.session_state.query_history = []
                    # Also clear the conversation memory
                    if 'conversation_memory' in st.session_state:
                        st.session_state.conversation_memory = None
                    st.rerun()
            with col2:
                if st.button("Download Conversation"):
                    # Create a downloadable version of the conversation
                    conversation_text = "# AlNu Health Conversation\n\n"
                    for i, exchange in enumerate(st.session_state.query_history):
                        conversation_text += f"## Exchange {i+1}\n"
                        conversation_text += f"**User:** {exchange['user']}\n\n"
                        conversation_text += f"**Assistant:** {exchange['assistant']}\n\n"
                    
                    # Provide the conversation as a download
                    st.download_button(
                        label="Download as Markdown",
                        data=conversation_text,
                        file_name="alnu_health_conversation.md",
                        mime="text/markdown"
                    )
            
            # Display conversation as a chat-like interface with a scrollable container
            chat_container = st.container()
            with chat_container:
                for i, exchange in enumerate(st.session_state.query_history):
                    # Create a more visual distinction between user and assistant
                    with st.chat_message("user"):
                        st.write(exchange['user'])
                    
                    with st.chat_message("assistant"):
                        st.write(exchange['assistant'])
        else:
            st.info("No conversation history yet. Ask a question to start.")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.success("Query history cleared")

if __name__ == "__main__":
    main()
