import os
import time
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any

from embeddings import CustomOpenAIEmbeddings
from config import OPENAI_API_KEY, CHROMA_PATH
from utils import verify_chroma_persistence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_chroma() -> Chroma:
    """Initialize ChromaDB with proper handling for conflicts"""
    # If we already have an instance in session state, return it
    if 'chroma_instance' in st.session_state and st.session_state.chroma_instance is not None:
        return st.session_state.chroma_instance
    
    # Check if ChromaDB directory exists - if it does, we'll try to load from it instead of resetting
    chroma_exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    logger.info(f"ChromaDB directory exists: {chroma_exists}")
    
    try:
        # Use our custom OpenAI embeddings implementation
        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
        logger.info("Successfully initialized CustomOpenAIEmbeddings")
        
        # Initialize Chroma with proper settings using the updated langchain-chroma package
        try:
            # Ensure the directory exists
            os.makedirs(CHROMA_PATH, exist_ok=True)
            
            # Create a simple Chroma instance without any client settings
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function,
                collection_name="documents"
            )
            
            # Store in session state
            st.session_state.chroma_instance = vectorstore
            st.session_state.db = vectorstore  # Also store as db for compatibility
            st.session_state.db_status = "Initialized"
            logger.info("Successfully initialized ChromaDB instance")
            
            # Return the instance
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            # Try a fallback approach
            try:
                # Direct import here to avoid any global configuration issues
                import chromadb
                
                # Create a new client without any proxy settings
                client = chromadb.PersistentClient(
                    path=CHROMA_PATH
                )
                
                # Create a new collection
                vectorstore = Chroma(
                    client=client,
                    collection_name="documents",
                    embedding_function=embedding_function
                )
                
                # Store in session state
                st.session_state.chroma_instance = vectorstore
                st.session_state.db_status = "Initialized (fallback)"
                
                logger.info("Successfully created ChromaDB instance using fallback method")
                return vectorstore
            except Exception as inner_e:
                logger.error(f"Error creating ChromaDB with fallback method: {str(inner_e)}")
                st.error(f"Failed to initialize vector database: {str(inner_e)}")
                st.session_state.db_status = "Failed to initialize"
                return None
    except Exception as e:
        logger.error(f"Error in initialize_chroma: {str(e)}")
        st.error(f"Failed to initialize vector database: {str(e)}")
        st.session_state.db_status = "Failed to initialize"
        return None

def create_vector_db(documents, update_existing=False):
    """Create or update vector database from documents"""
    try:
        # Start timer
        start_time = time.time()
        
        # Check if we have a valid ChromaDB instance
        if st.session_state.chroma_instance is None:
            st.session_state.chroma_instance = initialize_chroma()
            if st.session_state.chroma_instance is None:
                st.error("Failed to initialize vector database")
                return None
        
        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Setup progress tracking
        total_docs = len(documents)
        st.write(f"Creating vector database for {total_docs} documents...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each document
        for i, doc in enumerate(documents):
            # Update progress
            progress_percent = i / total_docs
            progress_bar.progress(progress_percent)
            status_text.write(f"Embedding document {i+1}/{total_docs}: {doc.get('name', 'unknown')}")
            
            # Skip if document is empty
            if not doc.get('content'):
                logger.warning(f"Empty content for document: {doc.get('name')}")
                continue
            
            # Create document chunks
            chunks = text_splitter.split_text(doc['content'])
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "source": doc.get('name', 'unknown'),
                    "chunk": i,
                    "document_id": doc.get('name', 'unknown'),
                }
                for i in range(len(chunks))
            ]
            
            # Add chunks to vector store
            st.session_state.chroma_instance.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.write(f"✅ Vector database creation complete! Processed {total_docs} documents.")
        
        # Persist changes
        st.session_state.chroma_instance.persist()
        
        # Update session state
        st.session_state.processed_docs = True
        st.session_state.last_processed_time = time.time()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Vector database created in {processing_time:.2f} seconds")
        
        return st.session_state.chroma_instance
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.error(f"Error creating vector database: {str(e)}")
        return None

def check_db_status():
    """Check ChromaDB status and reprocess if needed"""
    try:
        # Check if ChromaDB directory exists and has valid content
        if not verify_chroma_persistence(CHROMA_PATH):
            logger.warning("ChromaDB persistence verification failed")
            st.warning("Vector database needs to be reinitialized. Please process your documents again.")
            st.session_state.db_status = "Needs reinitialization"
            st.session_state.processed_docs = False
            return False
        
        # Check if we have a valid ChromaDB instance
        if 'chroma_instance' not in st.session_state or st.session_state.chroma_instance is None:
            logger.info("Initializing ChromaDB instance")
            st.session_state.chroma_instance = initialize_chroma()
            if st.session_state.chroma_instance is None:
                st.warning("Failed to initialize vector database. Please process your documents again.")
                st.session_state.db_status = "Failed to initialize"
                st.session_state.processed_docs = False
                return False
        
        # Update status
        st.session_state.db_status = "Ready"
        return True
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        st.error(f"Error checking database status: {str(e)}")
        st.session_state.db_status = "Error"
        return False
