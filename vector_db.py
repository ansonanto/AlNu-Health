import os
import time
import logging
import streamlit as st
import sqlite3
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Union

from embeddings import CustomOpenAIEmbeddings
from config import OPENAI_API_KEY, CHROMA_PATH
from utils import verify_chroma_persistence

# Check if we can use ChromaDB (requires SQLite >= 3.35.0)
SQLITE_VERSION = sqlite3.sqlite_version_info
MIN_SQLITE_VERSION = (3, 35, 0)
CAN_USE_CHROMA = SQLITE_VERSION >= MIN_SQLITE_VERSION

# Only import Chroma if SQLite version is compatible
if CAN_USE_CHROMA:
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_IMPORT_ERROR = None
    except ImportError as e:
        CHROMA_IMPORT_ERROR = str(e)
        CAN_USE_CHROMA = False
else:
    CHROMA_IMPORT_ERROR = f"SQLite version {SQLITE_VERSION} is not compatible with ChromaDB (requires >= {MIN_SQLITE_VERSION})"

# Import FAISS as a fallback vector database
try:
    from langchain_community.vectorstores import FAISS
    CAN_USE_FAISS = True
    FAISS_IMPORT_ERROR = None
except ImportError as e:
    CAN_USE_FAISS = False
    FAISS_IMPORT_ERROR = str(e)

# Define the vector database type to use
if CAN_USE_CHROMA:
    VECTOR_DB_TYPE = "chroma"
    logger.info("Using ChromaDB as vector database")
elif CAN_USE_FAISS:
    VECTOR_DB_TYPE = "faiss"
    logger.info("Using FAISS as fallback vector database (ChromaDB not available)")
else:
    VECTOR_DB_TYPE = None
    logger.warning("No compatible vector database available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define FAISS index path
FAISS_PATH = "./faiss_index"

def initialize_vector_db(reset_db=False) -> Optional[Any]:
    """Initialize vector database with proper handling for conflicts and version issues"""
    # If we already have an instance in session state and we're not resetting, return it
    if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
        return st.session_state.vector_db_instance
    
    # Use embedding function for either database type
    try:
        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
        logger.info("Successfully initialized CustomOpenAIEmbeddings")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        st.session_state.db_status = f"Error: Failed to initialize embeddings: {str(e)}"
        st.session_state.vector_db_instance = None
        st.session_state.db = None
        return None
    
    # Try ChromaDB first if available
    if VECTOR_DB_TYPE == "chroma":
        return initialize_chroma(reset_db, embedding_function)
    # Fall back to FAISS if ChromaDB isn't available
    elif VECTOR_DB_TYPE == "faiss":
        return initialize_faiss(reset_db, embedding_function)
    else:
        error_msg = "No compatible vector database available"
        logger.error(f"Failed to initialize vector database: {error_msg}")
        st.session_state.db_status = f"Error: {error_msg}"
        st.session_state.vector_db_instance = None
        st.session_state.db = None
        return None

def initialize_chroma(reset_db=False, embedding_function=None) -> Optional[Any]:
    """Initialize ChromaDB with proper handling for conflicts and SQLite version issues"""
    # If we already have an instance in session state and we're not resetting, return it
    if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None and st.session_state.vector_db_type == "chroma":
        return st.session_state.vector_db_instance
    
    # Check if ChromaDB can be used on this system
    if not CAN_USE_CHROMA:
        error_msg = CHROMA_IMPORT_ERROR or f"SQLite version {SQLITE_VERSION} is not compatible with ChromaDB"
        logger.error(f"Failed to initialize ChromaDB: {error_msg}")
        return None
    
    # If reset_db is True, delete the ChromaDB directory
    if reset_db and os.path.exists(CHROMA_PATH):
        import shutil
        try:
            logger.info(f"Resetting ChromaDB directory at {CHROMA_PATH}")
            # Create a backup directory
            backup_dir = f"{CHROMA_PATH}_backup_{int(time.time())}"
            if os.path.exists(CHROMA_PATH):
                shutil.copytree(CHROMA_PATH, backup_dir)
                logger.info(f"Created backup at {backup_dir}")
            # Remove the original directory
            shutil.rmtree(CHROMA_PATH)
            logger.info("ChromaDB directory reset successfully")
        except Exception as e:
            logger.error(f"Error resetting ChromaDB directory: {str(e)}")
    
    # Check if ChromaDB directory exists - if it does, we'll try to load from it instead of resetting
    chroma_exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    logger.info(f"ChromaDB directory exists: {chroma_exists}")
    
    try:
        # Use provided embedding function or create a new one
        if embedding_function is None:
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
            st.session_state.vector_db_instance = vectorstore
            st.session_state.vector_db_type = "chroma"
            st.session_state.db = vectorstore
            
            # Check if the database has documents
            try:
                count = vectorstore._collection.count()
                if count > 0:
                    st.session_state.processed_docs = True
                    st.session_state.db_status = "Healthy (ChromaDB - Loaded from disk)"
                    logger.info(f"Successfully loaded existing ChromaDB with {count} documents")
                else:
                    st.session_state.processed_docs = False
                    st.session_state.db_status = "Healthy (ChromaDB - Empty)"
                    logger.info("Successfully initialized empty ChromaDB")
            except Exception as e:
                logger.warning(f"Could not get document count: {str(e)}")
                st.session_state.db_status = "Healthy (ChromaDB - Unknown count)"
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in ChromaDB initialization process: {str(e)}")
        return None
        
def initialize_faiss(reset_db=False, embedding_function=None) -> Optional[Any]:
    """Initialize FAISS as a fallback vector database when ChromaDB isn't available"""
    # If we already have an instance in session state and we're not resetting, return it
    if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None and st.session_state.vector_db_type == "faiss":
        return st.session_state.vector_db_instance
    
    # Check if FAISS can be used
    if not CAN_USE_FAISS:
        error_msg = FAISS_IMPORT_ERROR or "FAISS is not available"
        logger.error(f"Failed to initialize FAISS: {error_msg}")
        st.session_state.db_status = f"Error: {error_msg}"
        st.session_state.vector_db_instance = None
        st.session_state.db = None
        return None
    
    # If reset_db is True, delete the FAISS index files
    if reset_db and os.path.exists(FAISS_PATH):
        import shutil
        try:
            logger.info(f"Resetting FAISS index at {FAISS_PATH}")
            # Create a backup directory
            backup_dir = f"{FAISS_PATH}_backup_{int(time.time())}"
            if os.path.exists(FAISS_PATH):
                shutil.copytree(FAISS_PATH, backup_dir)
                logger.info(f"Created backup at {backup_dir}")
            # Remove the original directory
            shutil.rmtree(FAISS_PATH)
            logger.info("FAISS index reset successfully")
        except Exception as e:
            logger.error(f"Error resetting FAISS index: {str(e)}")
    
    try:
        # Use provided embedding function or create a new one
        if embedding_function is None:
            embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
            logger.info("Successfully initialized CustomOpenAIEmbeddings for FAISS")
        
        # Check if FAISS index exists
        faiss_exists = os.path.exists(FAISS_PATH) and len(os.listdir(FAISS_PATH)) > 0
        logger.info(f"FAISS index exists: {faiss_exists}")
        
        # Initialize FAISS
        try:
            if faiss_exists:
                # Load existing index
                vectorstore = FAISS.load_local(FAISS_PATH, embedding_function, "documents")
                logger.info("Successfully loaded existing FAISS index")
                st.session_state.processed_docs = True
                st.session_state.db_status = "Healthy (FAISS - Loaded from disk)"
            else:
                # Create empty index (will be populated later)
                vectorstore = FAISS(embedding_function, [], [], "documents")
                os.makedirs(FAISS_PATH, exist_ok=True)
                vectorstore.save_local(FAISS_PATH)
                logger.info("Successfully initialized empty FAISS index")
                st.session_state.processed_docs = False
                st.session_state.db_status = "Healthy (FAISS - Empty)"
            
            # Store in session state
            st.session_state.vector_db_instance = vectorstore
            st.session_state.vector_db_type = "faiss"
            st.session_state.db = vectorstore
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            st.session_state.db_status = f"Error: {str(e)}"
            st.session_state.vector_db_instance = None
            st.session_state.db = None
            return None
            
    except Exception as e:
        logger.error(f"Error in FAISS initialization process: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        st.session_state.vector_db_instance = None
        st.session_state.db = None
        return None

def create_vector_db(documents, update_existing=False):
    """Create or update vector database from documents"""
    # Use the appropriate vector database based on what's available
    if VECTOR_DB_TYPE == "chroma":
        return create_chroma_db(documents, update_existing)
    elif VECTOR_DB_TYPE == "faiss":
        return create_faiss_db(documents, update_existing)
    else:
        logger.error("No compatible vector database available for creating/updating")
        st.session_state.db_status = "Error: No compatible vector database available"
        return None

def create_faiss_db(documents, update_existing=False):
    """Create or update FAISS vector database from documents"""
    try:
        # Start timer
        start_time = time.time()
        
        # Get embedding function
        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Check if we have an existing instance
        if update_existing and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None and st.session_state.vector_db_type == "faiss":
            # Update existing index
            vectorstore = st.session_state.vector_db_instance
            
            # Add documents to existing index
            vectorstore.add_documents(documents)
            
            # Save the updated index
            os.makedirs(FAISS_PATH, exist_ok=True)
            vectorstore.save_local(FAISS_PATH)
            
            # Update session state
            st.session_state.vector_db_instance = vectorstore
            st.session_state.db = vectorstore
            st.session_state.processed_docs = True
            st.session_state.db_status = "Healthy (FAISS - Updated)"
            
            # Log completion time
            end_time = time.time()
            logger.info(f"FAISS index updated in {end_time - start_time:.2f} seconds")
            
            return vectorstore
        else:
            # Create new index
            vectorstore = FAISS.from_documents(documents, embedding_function)
            
            # Save the index
            os.makedirs(FAISS_PATH, exist_ok=True)
            vectorstore.save_local(FAISS_PATH)
            
            # Update session state
            st.session_state.vector_db_instance = vectorstore
            st.session_state.vector_db_type = "faiss"
            st.session_state.db = vectorstore
            st.session_state.processed_docs = True
            st.session_state.db_status = "Healthy (FAISS - Created)"
            
            # Log completion time
            end_time = time.time()
            logger.info(f"FAISS index created in {end_time - start_time:.2f} seconds")
            
            return vectorstore
    except Exception as e:
        logger.error(f"Error creating/updating FAISS index: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return None

def create_chroma_db(documents, update_existing=False):
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
    """Check vector database status and reprocess if needed"""
    try:
        # Determine which vector database type to use
        if VECTOR_DB_TYPE == "chroma":
            # Check if ChromaDB directory exists and has valid content
            if not verify_chroma_persistence(CHROMA_PATH):
                logger.warning("ChromaDB persistence verification failed")
                st.warning("Vector database needs to be reinitialized. Please process your documents again.")
                st.session_state.db_status = "Needs reinitialization"
                st.session_state.processed_docs = False
                return False
        elif VECTOR_DB_TYPE == "faiss":
            # Check if FAISS index exists
            if not os.path.exists(FAISS_PATH) or len(os.listdir(FAISS_PATH)) == 0:
                logger.warning("FAISS index not found or empty")
                st.warning("Vector database needs to be initialized. Please process your documents.")
                st.session_state.db_status = "Needs initialization"
                st.session_state.processed_docs = False
                return False
        elif VECTOR_DB_TYPE is None:
            logger.warning("No compatible vector database available")
            st.warning("No compatible vector database available. Some features will be limited.")
            st.session_state.db_status = "Error: No compatible vector database available"
            st.session_state.processed_docs = False
            return False
        
        # Check if we have a valid vector database instance
        if 'vector_db_instance' not in st.session_state or st.session_state.vector_db_instance is None:
            logger.info(f"Initializing {VECTOR_DB_TYPE} instance")
            st.session_state.vector_db_instance = initialize_vector_db()
            if st.session_state.vector_db_instance is None:
                st.warning("Failed to initialize vector database. Please process your documents again.")
                st.session_state.db_status = "Failed to initialize"
                st.session_state.processed_docs = False
                return False
        
        # Update status if not already set
        if 'db_status' not in st.session_state or not st.session_state.db_status or st.session_state.db_status == "Error":
            st.session_state.db_status = f"Ready ({VECTOR_DB_TYPE.capitalize()})"
        
        return True
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False
