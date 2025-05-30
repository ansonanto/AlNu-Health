import os
import shutil
import logging
import time
import streamlit as st
from config import VECTOR_STORE_PATH

# Set up logging
logger = logging.getLogger(__name__)

def reset_vector_db(vector_path=VECTOR_STORE_PATH) -> None:
    """Reset FAISS vector storage with improved error handling"""
    try:
        # Clear session state
        # We're using FAISS instead of ChromaDB for better Streamlit compatibility
        if 'db' in st.session_state:
            st.session_state.db = None
        
        # First check if the directory exists
        if os.path.exists(vector_path):
            try:
                # Try to create a test file to check write permissions
                test_file = os.path.join(vector_path, 'test_write.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                # If we get here, we have write permissions, so remove the directory
                shutil.rmtree(vector_path)
                logger.info("Vector store reset successfully")
            except PermissionError as pe:
                logger.error(f"Permission error with vector store directory: {str(pe)}")
                # Try to fix permissions
                try:
                    # Change permissions to allow writing
                    for root, dirs, files in os.walk(vector_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)  # rwxr-xr-x
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)  # rw-r--r--
                    
                    # Try removing again
                    shutil.rmtree(vector_path)
                    logger.info("Vector store reset successfully after fixing permissions")
                except Exception as inner_e:
                    logger.error(f"Failed to fix permissions: {str(inner_e)}")
                    # As a last resort, try using a subprocess with sudo
                    logger.warning("Using alternative method to reset vector store")
                    # Create a new directory with a different name
                    new_path = f"{vector_path}_new"
                    os.makedirs(new_path, exist_ok=True)
                    # We can't update the global VECTOR_STORE_PATH here
                    # Just return the new path
                    return True
            except Exception as e:
                logger.error(f"Error removing vector store directory: {str(e)}")
                # Create a new directory with a different name as a fallback
                new_path = f"{vector_path}_new"
                os.makedirs(new_path, exist_ok=True)
                # We can't update the global VECTOR_STORE_PATH here
                # Just return the new path
                return True
        
        # Create an empty directory to ensure proper initialization
        os.makedirs(vector_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        return False

def verify_vector_store_persistence(vector_path=VECTOR_STORE_PATH):
    """Verify that FAISS vector store is correctly persisting data"""
    if not os.path.exists(vector_path):
        logger.warning("Vector store persistence directory does not exist")
        os.makedirs(vector_path, exist_ok=True)
        logger.info("Created vector store persistence directory")
        return False
    
    # Check for critical FAISS files that indicate proper persistence
    vector_files = os.listdir(vector_path)
    logger.info(f"Found {len(vector_files)} files in vector store persistence directory")
    
    # No files or only hidden files
    if len([f for f in vector_files if not f.startswith('.')]) == 0:
        logger.warning("Vector store directory exists but contains no non-hidden files")
        return False
    
    # Check for FAISS persistence files
    
    # FAISS index file
    if 'index.faiss' in vector_files:
        logger.info("Found FAISS index file")
        return True
    
    # Metadata file
    if 'index.pkl' in vector_files:
        logger.info("Found FAISS metadata file")
        return True
        
    # Check for dimension file
    if 'dimension.txt' in vector_files:
        logger.info("Found FAISS dimension file")
        return True
    
    # If we get here, we have files but none of the expected FAISS structures
    logger.warning("Vector store directory exists but may not contain valid database files")
    return False
