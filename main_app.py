# Must be the first import
import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="AlNu Health - RAG Document Search System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library imports
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
from PIL import Image
import openai

# Import configuration
from config import OPENAI_API_KEY, GEMINI_API_KEY, VECTOR_STORE_PATH

# Import utility functions
from document_processor import process_documents, PaperManager
from vector_db import initialize_vector_db, create_vector_db, check_db_status
from query_processor import query_documents
from pubmed_downloader import pubmed_downloader_ui
from prompt_evaluator import prompt_evaluator_ui
from food_analyzer import process_food_image, format_macro_display

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = OPENAI_API_KEY

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'new_documents' not in st.session_state:
        st.session_state.new_documents = []
    if 'last_processed_time' not in st.session_state:
        st.session_state.last_processed_time = None
    if 'db_status' not in st.session_state:
        st.session_state.db_status = "Not initialized"
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'accuracy_percentage' not in st.session_state:
        st.session_state.accuracy_percentage = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Document Management"

# Initialize session state variables if they don't exist
initialize_session_state()

# Try to initialize the vector database on startup
try:
    # Check for existing vector store at startup
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss_index.bin")):
        logger.info("Found existing vector database, attempting to load")
        # Initialize the database
        db = initialize_vector_db()
        if db is not None:
            st.session_state.db = db
            st.session_state.processed_docs = True
            st.session_state.db_status = "Vector database loaded successfully"
        else:
            st.session_state.processed_docs = False
            st.session_state.db_status = "Failed to load vector database"
    else:
        logger.info("No existing vector database found or verification failed")
except Exception as e:
    logger.error(f"Error initializing vector database on startup: {str(e)}")
    # Don't raise the exception, just log it

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

def save_query_history():
    """Save query history to disk"""
    history_path = os.path.join(VECTOR_STORE_PATH, "query_history.json")
    try:
        if 'query_history' in st.session_state:
            with open(history_path, 'w') as f:
                json.dump(st.session_state.query_history, f)
            logger.info("Query history saved successfully")
    except Exception as e:
        logger.error(f"Error saving query history: {str(e)}")

def load_query_history():
    """Load query history from disk"""
    history_path = os.path.join(VECTOR_STORE_PATH, "query_history.json")
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            st.session_state.query_history = history
            logger.info("Query history loaded successfully")
    except Exception as e:
        logger.error(f"Error loading query history: {str(e)}")
        st.session_state.query_history = []

def main():
    """Main application UI"""
    # Initialize session state
    initialize_session_state()
    
    # Load query history
    if 'query_history' not in st.session_state:
        load_query_history()
    
    # Display header
    st.title("🏥 AlNu Health - Medical Research RAG System")
    
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
            # Initialize vector store with reset flag
            if initialize_vector_db(reset_db=True):
                # Clear all relevant session state except query history
                st.session_state.db = None
                st.session_state.processed_docs = False
                st.session_state.vector_store = None
                st.session_state.documents = []
                st.session_state.new_documents = []
                st.session_state.last_processed_time = None
                st.session_state.db_status = "Not initialized"
                
                st.success("Vector database reset successfully")
                st.experimental_rerun()
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
    
    # Create tabs for text and image input
    input_tabs = st.tabs(["Text Query", "Food Image"])
    
    with input_tabs[0]:  # Text Query tab
        st.subheader("Ask a Medical Research Question")
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # First check if we need to initialize the database
            if st.session_state.db is None:
                # Try to initialize the database if it exists
                if os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss_index.bin")):
                    logger.info("Attempting to initialize vector database for search")
                    db = initialize_vector_db()
                    if db is not None:
                        st.session_state.db = db
                        st.session_state.processed_docs = True
                        st.session_state.db_status = "Vector database loaded successfully"
                    else:
                        st.session_state.processed_docs = False
                        st.session_state.db_status = "Failed to load vector database"
            
            # Check if documents have been processed
            if not st.session_state.processed_docs or st.session_state.db is None:
                st.warning("Please process documents first in the Document Management tab")
                # Add a button to process documents directly from this tab
                if st.button("Process Documents Now"):
                    with st.spinner("Processing documents..."):
                        documents, new_docs = process_documents()
                        if documents:
                            st.session_state.documents = documents
                            st.session_state.new_documents = new_docs
                            db = create_vector_db(documents)
                            if db:
                                st.session_state.db = db
                                st.success(f"Successfully processed {len(documents)} documents")
                                st.rerun()  # Refresh the page
                            else:
                                st.error("Failed to create vector database")
                        else:
                            st.warning("No documents to process")
                return
            
            # Query input
            query = st.text_area("Enter your question:", height=100)
            
            # Query button for text search
            if st.button("Search", key="text_search"):
                if query:
                    with st.spinner("Searching..."):
                        # Get conversation history
                        conversation_history = st.session_state.query_history[-5:] if st.session_state.query_history else []
                        
                        # Query documents
                        results = query_documents(query, st.session_state.db, conversation_history)
                        
                        # Store results
                        st.session_state.search_results = results
    
    with input_tabs[1]:  # Food Image tab
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h3 style="color: #4CAF50; margin-top: 0;">🍽️ Food Image Analysis</h3>
            <p style="color: white;">Upload an image of your food to get detailed nutritional information using AI.</p>
        </div>
        """, unsafe_allow_html=True)
            
        # Upload image
        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Create columns for better layout
            img_col, info_col = st.columns([1, 1])
            
            with img_col:
                # Display the uploaded image with a border
                image = Image.open(uploaded_file)
                st.markdown("""
                <div style="padding: 5px; border: 2px solid #4CAF50; border-radius: 10px; display: inline-block;">
                """, unsafe_allow_html=True)
                st.image(image, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with info_col:
                st.markdown("""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; height: 100%;">
                    <h4 style="color: #42A5F5; margin-top: 0;">How it works:</h4>
                    <ul style="color: white;">
                        <li>AI first verifies if the image contains food</li>
                        <li>Then extracts detailed nutritional information</li>
                        <li>Results are displayed with visual indicators</li>
                        <li>Analysis is saved in your query history</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Process button with improved styling
            st.markdown("""
            <style>
            div.stButton > button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
            }
            div.stButton > button:hover {
                background-color: #45a049;
            }
            </style>
            """, unsafe_allow_html=True)
            if st.button("🔍 Analyze Food", key="analyze_food"):
                with st.spinner("Analyzing your food image..."):
                    # Process the food image
                    success, message, macros = process_food_image(uploaded_file)
                    if success:
                        # Format and display the results
                        format_macro_display(macros)
                        
                        # Add to query history and save
                        st.session_state.query_history.append({
                            "user": "[Food Image Analysis]",
                            "assistant": message,
                            "food_image": True,
                            "macros": macros
                        })
                        save_query_history()
                    else:
                        st.error(message)

    # Display results (outside both tabs)
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
                # Customize display for food image analysis
                if exchange.get('food_image', False):
                    # Create a more visually appealing expander for food analysis
                    with st.expander(f"🍽️ Food Analysis: {exchange.get('macros', {}).get('food_name', 'Unknown Food')}"):
                        if 'macros' in exchange:
                            # Display formatted macros with native Streamlit components
                            format_macro_display(exchange['macros'])
                        else:
                            st.write(exchange["assistant"])
                else:
                    # Regular text query display
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
            save_query_history()
            st.success("Query history cleared")

if __name__ == "__main__":
    main()
