"""
LlamaIndex-based vector database implementation for AlNu Health.
This provides a more compatible alternative to ChromaDB/FAISS for Streamlit deployment.
"""
import os
import time
import logging
import streamlit as st
from typing import List, Dict, Any, Optional, Union

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Local imports
from config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
STORAGE_DIR = "./llama_index_storage"
FAISS_INDEX_FILE = os.path.join(STORAGE_DIR, "faiss_index.bin")

# Initialize settings
def init_settings():
    """Initialize LlamaIndex settings"""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is missing")
        raise ValueError("OpenAI API key is required for embeddings")
    
    # Set up the embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
        dimensions=1536
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 200
    
    # Set up node parser for text splitting
    Settings.node_parser = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    logger.info("LlamaIndex settings initialized")
    return True

def initialize_vector_db(reset_db=False) -> Optional[Any]:
    """Initialize vector database with LlamaIndex"""
    try:
        # Check if we already have an instance in session state
        if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
            return st.session_state.vector_db_instance
        
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            error_msg = "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets."
            logger.error(error_msg)
            st.session_state.db_status = f"Error: {error_msg}"
            return None
        
        # Initialize settings
        init_settings()
        
        # Create storage directory if it doesn't exist
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        # Check if we need to reset the database
        if reset_db and os.path.exists(FAISS_INDEX_FILE):
            logger.info(f"Resetting vector database at {FAISS_INDEX_FILE}")
            try:
                os.remove(FAISS_INDEX_FILE)
                logger.info("Vector database reset successfully")
            except Exception as e:
                logger.error(f"Error resetting vector database: {str(e)}")
        
        # Create a new empty index
        try:
            # Set up FAISS vector store
            vector_store = FaissVectorStore(dim=1536)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create an empty index
            index = VectorStoreIndex([], storage_context=storage_context)
            
            # Store in session state
            st.session_state.vector_db_instance = index
            st.session_state.vector_db_type = "llama_index"
            st.session_state.db = index
            st.session_state.processed_docs = False
            st.session_state.db_status = "Healthy (LlamaIndex - Empty)"
            
            logger.info("Successfully initialized empty LlamaIndex vector database")
            return index
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            st.session_state.db_status = f"Error: {str(e)}"
            return None
            
    except Exception as e:
        logger.error(f"Error in vector database initialization: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return None

def create_vector_db(documents, update_existing=False):
    """Create or update vector database from documents"""
    try:
        # Start timer
        start_time = time.time()
        
        # Log document count
        logger.info(f"Creating vector database with {len(documents)} documents")
        
        # Convert documents to LlamaIndex format
        llama_docs = []
        for doc in documents:
            try:
                if not doc.get("content"):
                    logger.warning(f"Empty content for document: {doc.get('name')}")
                    continue
                
                # Create LlamaIndex Document
                llama_doc = Document(
                    text=doc["content"],
                    metadata={
                        "name": doc.get("name", "unknown"),
                        "path": doc.get("path", ""),
                        "source": doc.get("name", "unknown")
                    }
                )
                llama_docs.append(llama_doc)
            except Exception as doc_e:
                logger.error(f"Error converting document: {str(doc_e)}")
        
        logger.info(f"Converted {len(llama_docs)} documents to LlamaIndex format")
        
        # Initialize settings if not already done
        init_settings()
        
        # Get existing index or create new one
        if update_existing and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
            # Update existing index
            index = st.session_state.vector_db_instance
            for doc in llama_docs:
                index.insert(doc)
            
            logger.info(f"Updated existing index with {len(llama_docs)} documents")
        else:
            # Create new index
            vector_store = FaissVectorStore(dim=1536)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(llama_docs, storage_context=storage_context)
            logger.info(f"Created new index with {len(llama_docs)} documents")
        
        # Update session state
        st.session_state.vector_db_instance = index
        st.session_state.vector_db_type = "llama_index"
        st.session_state.db = index
        st.session_state.processed_docs = True
        st.session_state.db_status = "Healthy (LlamaIndex)"
        
        # Log completion time
        end_time = time.time()
        logger.info(f"Vector database created in {end_time - start_time:.2f} seconds")
        
        return index
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.session_state.db_status = f"Error: Failed to create vector database: {str(e)}"
        return None

def check_db_status():
    """Check vector database status"""
    try:
        # Check if we have a valid vector database instance
        if 'vector_db_instance' not in st.session_state or st.session_state.vector_db_instance is None:
            logger.info("Initializing vector database")
            st.session_state.vector_db_instance = initialize_vector_db()
            if st.session_state.vector_db_instance is None:
                st.warning("Failed to initialize vector database. Please check your API keys.")
                st.session_state.db_status = "Failed to initialize"
                st.session_state.processed_docs = False
                return False
        
        # Update status if not already set
        if 'db_status' not in st.session_state or not st.session_state.db_status:
            st.session_state.db_status = "Ready (LlamaIndex)"
        
        return True
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False

def query_index(query_text, top_k=5):
    """Query the vector database index"""
    try:
        if 'vector_db_instance' not in st.session_state or st.session_state.vector_db_instance is None:
            logger.error("No vector database instance available")
            return []
        
        # Get the index
        index = st.session_state.vector_db_instance
        
        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        
        # Execute query
        response = query_engine.query(query_text)
        
        # Return source nodes
        return response.source_nodes
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        return []
