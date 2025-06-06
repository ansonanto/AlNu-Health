import os
import time
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document

# Import the direct Gemini embeddings implementation
from gemini_embeddings import GeminiEmbeddings
from config import GEMINI_API_KEY, VECTOR_STORE_PATH, EMBEDDING_DIMENSION
from faiss_store import FAISSVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_vector_store():
    """Delete the vector store files"""
    try:
        # Use consistent file names
        index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
        metadata_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")
        dimension_path = os.path.join(VECTOR_STORE_PATH, "dimension.txt")
        
        # Delete files if they exist
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        if os.path.exists(dimension_path):
            os.remove(dimension_path)
            
        logger.info("Vector store files deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting vector store files: {str(e)}")
        return False

def initialize_vector_db(reset_db=False) -> Optional[FAISSVectorStore]:
    """Initialize FAISS vector store"""
    # If resetting, delete existing files
    if reset_db:
        if not delete_vector_store():
            return None
    
    # If we already have an instance in session state and we're not resetting, return it
    if not reset_db and 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        return st.session_state.vector_store
    
    try:
        # Initialize embedding function with the correct dimension
        # Use the direct Gemini embeddings implementation
        embedding_function = GeminiEmbeddings(
            api_key=GEMINI_API_KEY,
            output_dim=EMBEDDING_DIMENSION
        )
        
        # Create vector store
        vector_store = FAISSVectorStore(embedding_function, VECTOR_STORE_PATH)
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.db_status = "Vector store loaded successfully"
        
        # Set the processed_docs flag to True so the UI will display the documents
        st.session_state.processed_docs = True
        
        return vector_store
    except Exception as e:
        error_msg = f"Failed to initialize vector store: {str(e)}"
        logger.error(error_msg)
        st.session_state.db_status = f"Error: {error_msg}"
        st.session_state.vector_store = None
        return None

def create_vector_db(documents: List[Dict[str, Any]], update_existing: bool = False) -> bool:
    """Create or update vector database from documents"""
    print("create_vector_db called with", len(documents), "documents")
    for doc in documents[:5]:
        print("Doc name:", doc.get('name'), "Content length:", len(doc.get('content', '')))
    try:
        # Initialize vector store
        vector_store = initialize_vector_db(reset_db=update_existing)
        if not vector_store:
            return False

        # Convert documents to LangChain Document format
        doc_objects = []
        for doc in documents:
            metadata = {
                'source': doc.get('name', 'unknown'),
                'document_id': doc.get('name', 'unknown')
            }
            metadata.update(doc.get('metadata', {}))
            
            doc_objects.append(Document(
                page_content=doc.get('content', ''),
                metadata=metadata
            ))

        print("Calling add_documents with", len(doc_objects), "documents")
        vector_store.add_documents(doc_objects, update_existing=update_existing)

        # Debug print: list files in vector DB directory after processing
        print("Files in vector DB directory after processing:", os.listdir(VECTOR_STORE_PATH))

        # Update session state
        st.session_state.db_status = "Vector database created successfully"
        return True

    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False

def check_db_status() -> bool:
    """Check vector store status"""
    try:
        # Check if vector store exists - use consistent file name
        if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            logger.info("No existing vector store found")
            return False

        # Try to initialize vector store
        if initialize_vector_db():
            st.session_state.db_status = "Vector store loaded successfully"
            return True
        else:
            logger.warning("Vector store initialization failed")
            return False

    except Exception as e:
        logger.error(f"Error checking vector store status: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False
