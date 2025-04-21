import os
import time
import logging
import streamlit as st
import sqlite3

# Set up logging
logger = logging.getLogger(__name__)
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
    if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
        return st.session_state.vector_db_instance
    
    # Check if OpenAI API key is available
    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        error_msg = "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets."
        logger.error(error_msg)
        st.session_state.db_status = f"Error: {error_msg}"
        return None
    
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
                # Try to load existing index with error handling
                try:
                    # First try standard loading method
                    vectorstore = FAISS.load_local(FAISS_PATH, embedding_function, "documents")
                    logger.info("Successfully loaded existing FAISS index using standard method")
                    st.session_state.processed_docs = True
                    st.session_state.db_status = "Healthy (FAISS - Loaded from disk)"
                except Exception as e:
                    logger.warning(f"Standard FAISS loading failed: {str(e)}")
                    
                    # Check for our custom format
                    custom_marker = os.path.join(FAISS_PATH, 'custom_format.txt')
                    index_path = os.path.join(FAISS_PATH, 'index.faiss')
                    docstore_path = os.path.join(FAISS_PATH, 'docstore.json')
                    
                    if os.path.exists(custom_marker) and os.path.exists(index_path) and os.path.exists(docstore_path):
                        try:
                            # Load the FAISS index directly
                            import faiss
                            faiss_index = faiss.read_index(index_path)
                            
                            # Load the document store
                            import json
                            from langchain.schema import Document
                            from langchain.docstore.in_memory import InMemoryDocstore
                            
                            with open(docstore_path, 'r') as f:
                                docstore_data = json.load(f)
                            
                            # Convert back to Document objects
                            docstore_dict = {}
                            for k, doc_data in docstore_data.items():
                                doc = Document(
                                    page_content=doc_data['page_content'],
                                    metadata=doc_data['metadata']
                                )
                                docstore_dict[k] = doc
                            
                            docstore = InMemoryDocstore(docstore_dict)
                            
                            # Recreate the FAISS vectorstore
                            vectorstore = FAISS(embedding_function, faiss_index, docstore, "documents")
                            
                            logger.info("Successfully loaded existing FAISS index using custom format")
                            st.session_state.processed_docs = True
                            st.session_state.db_status = "Healthy (FAISS - Loaded from custom format)"
                        except Exception as inner_e:
                            logger.error(f"Custom format loading failed: {str(inner_e)}")
                            # Create empty index as fallback
                            vectorstore = FAISS(embedding_function, [], [], "documents")
                            logger.warning("Created empty FAISS index as fallback")
                            st.session_state.processed_docs = False
                            st.session_state.db_status = "Healthy (FAISS - Empty, loading failed)"
                    else:
                        # Create empty index as fallback
                        vectorstore = FAISS(embedding_function, [], [], "documents")
                        logger.warning("Created empty FAISS index as fallback (no pickle found)")
                        st.session_state.processed_docs = False
                        st.session_state.db_status = "Healthy (FAISS - Empty, loading failed)"
            else:
                # Create empty index (will be populated later)
                vectorstore = FAISS(embedding_function, [], [], "documents")
                os.makedirs(FAISS_PATH, exist_ok=True)
                
                # Try to save the empty index with error handling
                try:
                    vectorstore.save_local(FAISS_PATH)
                    logger.info("Successfully initialized and saved empty FAISS index")
                except TypeError as e:
                    # Handle the argument error for older FAISS versions
                    logger.warning(f"Using alternative save method for empty index due to: {str(e)}")
                    try:
                        # Try saving with a different method - save only the index and embeddings
                        # Extract just the FAISS index and embeddings
                        faiss_index = vectorstore.index
                        docstore = vectorstore.docstore
                        
                        # Save the index directly using FAISS methods
                        import faiss
                        index_path = os.path.join(FAISS_PATH, 'index.faiss')
                        faiss.write_index(faiss_index, index_path)
                        
                        # Save metadata separately
                        import json
                        with open(os.path.join(FAISS_PATH, 'docstore.json'), 'w') as f:
                            # Convert docstore to a serializable format
                            docstore_data = {}
                            for k, doc in docstore._dict.items():
                                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                                    docstore_data[k] = {
                                        'page_content': doc.page_content,
                                        'metadata': doc.metadata
                                    }
                            json.dump(docstore_data, f)
                        
                        # Create a marker file to indicate this is our custom format
                        with open(os.path.join(FAISS_PATH, 'custom_format.txt'), 'w') as f:
                            f.write('This index was saved in a custom format to avoid pickling issues.')
                            
                        logger.info("Successfully saved empty FAISS index using custom method")
                    except Exception as inner_e:
                        logger.error(f"Failed to save empty FAISS index with alternative method: {str(inner_e)}")
                        # Continue anyway since we have the in-memory instance
                
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
    try:
        # Log the document count and types for debugging
        logger.info(f"Creating vector database with {len(documents)} documents")
        logger.info(f"Document types: {[type(doc) for doc in documents[:3]]}")
        logger.info(f"Document keys: {[doc.keys() for doc in documents[:3]]}")
        
        # Use the appropriate vector database based on what's available
        if VECTOR_DB_TYPE == "chroma":
            logger.info("Using ChromaDB for vector database creation")
            return create_chroma_db(documents, update_existing)
        elif VECTOR_DB_TYPE == "faiss":
            logger.info("Using FAISS for vector database creation")
            return create_faiss_db(documents, update_existing)
        else:
            logger.error("No compatible vector database available for creating/updating")
            st.session_state.db_status = "Error: No compatible vector database available"
            return None
    except Exception as e:
        logger.error(f"Error in create_vector_db: {str(e)}")
        st.session_state.db_status = f"Error: Failed to create vector database: {str(e)}"
        return None

def create_faiss_db(documents, update_existing=False):
    """Create or update FAISS vector database from documents"""
    try:
        # Start timer
        start_time = time.time()
        
        # Convert documents to the format expected by FAISS
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create a text splitter for chunking large documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Smaller chunks to avoid token limits
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Process and chunk documents
        langchain_docs = []
        for doc in documents:
            try:
                if not doc.get("content"):
                    logger.warning(f"Empty content for document: {doc.get('name')}")
                    continue
                    
                # Split the document into smaller chunks
                doc_content = doc["content"]
                chunks = text_splitter.split_text(doc_content)
                logger.info(f"Split document '{doc.get('name')}' into {len(chunks)} chunks")
                
                # Create a Document for each chunk with metadata
                for i, chunk in enumerate(chunks):
                    langchain_doc = Document(
                        page_content=chunk,
                        metadata={
                            "name": doc.get("name", "unknown"),
                            "path": doc.get("path", ""),
                            "chunk": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    langchain_docs.append(langchain_doc)
            except Exception as doc_e:
                logger.error(f"Error converting document to langchain format: {str(doc_e)}")
                logger.error(f"Document keys: {doc.keys() if isinstance(doc, dict) else 'Not a dict'}")
        
        logger.info(f"Converted {len(documents)} documents into {len(langchain_docs)} chunks for embedding")
        
        # Get embedding function
        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Check if we have an existing instance
        if update_existing and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None and st.session_state.vector_db_type == "faiss":
            # Update existing index
            vectorstore = st.session_state.vector_db_instance
            
            # Add documents to existing index
            vectorstore.add_documents(langchain_docs)
            
            # Save the updated index with error handling
            os.makedirs(FAISS_PATH, exist_ok=True)
            try:
                vectorstore.save_local(FAISS_PATH)
            except TypeError as e:
                # Handle the argument error for older FAISS versions
                logger.warning(f"Using alternative save method due to: {str(e)}")
                try:
                    # Try saving with a different method
                    import pickle
                    with open(os.path.join(FAISS_PATH, 'index.pickle'), 'wb') as f:
                        pickle.dump(vectorstore, f)
                    logger.info("Successfully saved FAISS index using pickle method")
                except Exception as inner_e:
                    logger.error(f"Failed to save FAISS index with alternative method: {str(inner_e)}")
            
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
            vectorstore = FAISS.from_documents(langchain_docs, embedding_function)
            
            # Save the index with error handling
            os.makedirs(FAISS_PATH, exist_ok=True)
            try:
                vectorstore.save_local(FAISS_PATH)
            except TypeError as e:
                # Handle the argument error for older FAISS versions
                logger.warning(f"Using alternative save method due to: {str(e)}")
                try:
                    # Extract just the FAISS index and embeddings
                    faiss_index = vectorstore.index
                    docstore = vectorstore.docstore
                    
                    # Save the index directly using FAISS methods
                    import faiss
                    index_path = os.path.join(FAISS_PATH, 'index.faiss')
                    faiss.write_index(faiss_index, index_path)
                    
                    # Save metadata separately
                    import json
                    with open(os.path.join(FAISS_PATH, 'docstore.json'), 'w') as f:
                        # Convert docstore to a serializable format
                        docstore_data = {}
                        for k, doc in docstore._dict.items():
                            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                                docstore_data[k] = {
                                    'page_content': doc.page_content,
                                    'metadata': doc.metadata
                                }
                        json.dump(docstore_data, f)
                    
                    # Create a marker file to indicate this is our custom format
                    with open(os.path.join(FAISS_PATH, 'custom_format.txt'), 'w') as f:
                        f.write('This index was saved in a custom format to avoid pickling issues.')
                        
                    logger.info("Successfully saved FAISS index using custom method")
                except Exception as inner_e:
                    logger.error(f"Failed to save FAISS index with alternative method: {str(inner_e)}")
            
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
        
        # Convert documents to the format expected by ChromaDB
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create a text splitter for chunking large documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Smaller chunks to avoid token limits
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Process and chunk documents
        langchain_docs = []
        for doc in documents:
            try:
                if not doc.get("content"):
                    logger.warning(f"Empty content for document: {doc.get('name')}")
                    continue
                    
                # Split the document into smaller chunks
                doc_content = doc["content"]
                chunks = text_splitter.split_text(doc_content)
                logger.info(f"Split document '{doc.get('name')}' into {len(chunks)} chunks")
                
                # Create a Document for each chunk with metadata
                for i, chunk in enumerate(chunks):
                    langchain_doc = Document(
                        page_content=chunk,
                        metadata={
                            "name": doc.get("name", "unknown"),
                            "path": doc.get("path", ""),
                            "chunk": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    langchain_docs.append(langchain_doc)
            except Exception as doc_e:
                logger.error(f"Error converting document to langchain format: {str(doc_e)}")
                logger.error(f"Document keys: {doc.keys() if isinstance(doc, dict) else 'Not a dict'}")
        
        logger.info(f"Converted {len(documents)} documents into {len(langchain_docs)} chunks for ChromaDB")
        
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
        
        # Use the Chroma add_documents method directly with the langchain_docs
        try:
            # Get the ChromaDB instance
            chroma_db = st.session_state.chroma_instance
            
            # Add documents to ChromaDB
            logger.info(f"Adding {len(langchain_docs)} documents to ChromaDB")
            chroma_db.add_documents(langchain_docs)
            
            # Update status
            st.success(f"Successfully added {len(langchain_docs)} documents to vector database")
            st.session_state.processed_docs = True
            st.session_state.db_status = "Healthy (ChromaDB)"
            
            # Log completion time
            end_time = time.time()
            logger.info(f"ChromaDB updated in {end_time - start_time:.2f} seconds")
            
            return chroma_db
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            st.error(f"Error adding documents to vector database: {str(e)}")
            st.session_state.db_status = f"Error: {str(e)}"
            return None
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
