"""
LangChain-based vector database implementation for AlNu Health.
Uses ChromaDB as the vector store.
"""
import os
import time
import logging
import streamlit as st
import sqlite3
from typing import List, Dict, Any, Optional, Union

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document as LangchainDocument

# Local imports
from embeddings import CustomOpenAIEmbeddings
from config import OPENAI_API_KEY, CHROMA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we can use ChromaDB (requires SQLite >= 3.35.0)
SQLITE_VERSION = sqlite3.sqlite_version_info
MIN_SQLITE_VERSION = (3, 35, 0)
CAN_USE_CHROMA = SQLITE_VERSION >= MIN_SQLITE_VERSION

def initialize_vector_db(reset_db=False) -> Optional[Any]:
    """Initialize vector database with LangChain and ChromaDB"""
    try:
        # Check SQLite version compatibility
        if not CAN_USE_CHROMA:
            error_msg = f"SQLite version {SQLITE_VERSION} is not compatible with ChromaDB (requires >= {MIN_SQLITE_VERSION})"
            logger.error(error_msg)
            st.session_state.db_status = f"Error: {error_msg}"
            return None
        
        # Check if we already have an instance in session state
        if not reset_db and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
            return st.session_state.vector_db_instance
        
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            error_msg = "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets."
            logger.error(error_msg)
            st.session_state.db_status = f"Error: {error_msg}"
            return None
        
        # Create embedding function
        embedding_function = CustomOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create storage directory if it doesn't exist
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Check if we need to reset the database
        if reset_db and os.path.exists(CHROMA_PATH):
            logger.info(f"Resetting vector database at {CHROMA_PATH}")
            try:
                import shutil
                # Remove all files except .gitkeep
                for item in os.listdir(CHROMA_PATH):
                    if item != '.gitkeep':
                        item_path = os.path.join(CHROMA_PATH, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                logger.info("Vector database reset successfully")
            except Exception as e:
                logger.error(f"Error resetting vector database: {str(e)}")
        
        # Initialize ChromaDB
        try:
            logger.info(f"Initializing ChromaDB at {CHROMA_PATH}")
            # Initialize ChromaDB with client settings to avoid tenant issues
            from chromadb.config import Settings
            from chromadb import PersistentClient
            
            # Create client with explicit settings
            client = PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Create or get collection
            collection_name = "alnu_health"
            try:
                # Try to get existing collection
                collection = client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except Exception as e:
                # Create new collection if it doesn't exist
                logger.info(f"Creating new collection: {collection_name}")
                collection = client.create_collection(collection_name)
            
            # Initialize Chroma with the collection
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=CHROMA_PATH
            )
            
            # Store in session state
            st.session_state.vector_db_instance = vectorstore
            st.session_state.vector_db_type = "langchain_chroma"
            st.session_state.db = vectorstore
            
            # Check if there are any documents in the database
            collection = vectorstore._collection
            count = collection.count()
            
            if count > 0:
                st.session_state.processed_docs = True
                st.session_state.db_status = f"Healthy (LangChain ChromaDB - {count} documents)"
                logger.info(f"Successfully loaded existing ChromaDB with {count} documents")
            else:
                st.session_state.processed_docs = False
                st.session_state.db_status = "Healthy (LangChain ChromaDB - Empty)"
                logger.info("Successfully initialized empty ChromaDB")
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            st.session_state.db_status = f"Error: {str(e)}"
            return None
            
    except Exception as e:
        logger.error(f"Error in vector database initialization: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return None

def create_vector_db(documents, update_existing=False):
    """Create or update vector database from documents"""
    try:
        # Check SQLite version compatibility
        if not CAN_USE_CHROMA:
            error_msg = f"SQLite version {SQLITE_VERSION} is not compatible with ChromaDB (requires >= {MIN_SQLITE_VERSION})"
            logger.error(error_msg)
            st.session_state.db_status = f"Error: {error_msg}"
            return None
        
        # Start timer
        start_time = time.time()
        
        # Log document count
        logger.info(f"Creating vector database with {len(documents)} documents")
        
        # Create progress tracking elements in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.write("Initializing vector database...")
        
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            error_msg = "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets."
            logger.error(error_msg)
            st.session_state.db_status = f"Error: {error_msg}"
            progress_bar.progress(1.0)
            status_text.write("❌ Error: OpenAI API key is missing")
            return None
        
        # Create embedding function
        embedding_function = CustomOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Process documents in batches to avoid memory issues
        batch_size = 5
        total_docs = len(documents)
        successful_docs = 0
        total_chunks = 0
        processed_chunks = 0
        
        # Initialize or get existing ChromaDB
        if update_existing and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
            vectorstore = st.session_state.vector_db_instance
            logger.info("Using existing ChromaDB instance")
        else:
            # Initialize a new ChromaDB instance
            progress_bar.progress(0.1)
            status_text.write("Initializing new ChromaDB instance...")
            
            # Clear existing data if not updating
            if not update_existing:
                # Reset the database
                try:
                    import shutil
                    # Remove all files except .gitkeep
                    for item in os.listdir(CHROMA_PATH):
                        if item != '.gitkeep':
                            item_path = os.path.join(CHROMA_PATH, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                    logger.info("Cleared existing ChromaDB data")
                except Exception as e:
                    logger.error(f"Error clearing ChromaDB data: {str(e)}")
            
            # Initialize ChromaDB with client settings to avoid tenant issues
            from chromadb.config import Settings
            from chromadb import PersistentClient
            
            # Create client with explicit settings
            client = PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Create or get collection
            collection_name = "alnu_health"
            try:
                # Try to get existing collection
                collection = client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except Exception as e:
                # Create new collection if it doesn't exist
                logger.info(f"Creating new collection: {collection_name}")
                collection = client.create_collection(collection_name)
            
            # Initialize Chroma with the collection
            vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=CHROMA_PATH
            )
            logger.info("Initialized new ChromaDB instance")
        
        # Process documents in batches
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = documents[batch_start:batch_end]
            
            try:
                batch_docs = []
                batch_chunks_count = 0
                
                # Update status
                batch_progress = batch_start / total_docs
                progress_bar.progress(0.1 + batch_progress * 0.4)  # 10-50% progress during chunking
                status_text.write(f"Processing batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}...")
                
                for doc in batch:
                    if not doc.get("content"):
                        logger.warning(f"Empty content for document: {doc.get('name')}")
                        continue
                    
                    # Split text into chunks to avoid token limits
                    try:
                        chunks = text_splitter.split_text(doc["content"])
                        batch_chunks_count += len(chunks)
                        
                        # Update status with document name
                        status_text.write(f"Chunking document: {doc.get('name')} into {len(chunks)} parts...")
                        
                        for i, chunk in enumerate(chunks):
                            batch_docs.append(
                                LangchainDocument(
                                    page_content=chunk,
                                    metadata={
                                        "source": doc.get("name", "unknown"),
                                        "chunk": i,
                                        "total_chunks": len(chunks),
                                        "name": doc.get("name", "unknown"),
                                        "path": doc.get("path", "")
                                    }
                                )
                            )
                    except Exception as chunk_e:
                        logger.error(f"Error chunking document {doc.get('name')}: {str(chunk_e)}")
                
                # Update total chunks count
                total_chunks += batch_chunks_count
                
                if batch_docs:
                    # Update status for embedding
                    progress_bar.progress(0.5 + (batch_start / total_docs) * 0.4)  # 50-90% progress during embedding
                    status_text.write(f"Embedding {len(batch_docs)} chunks from batch {batch_start//batch_size + 1}...")
                    
                    # Add documents to ChromaDB
                    vectorstore.add_documents(batch_docs)
                    
                    # Update processed chunks count
                    processed_chunks += len(batch_docs)
                    successful_docs += len(batch_docs)
                    
                    # Persist after each batch
                    status_text.write(f"Persisting ChromaDB after batch {batch_start//batch_size + 1}...")
                    vectorstore.persist()
                    
                    logger.info(f"Successfully processed batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_start//batch_size + 1}: {str(e)}")
                # Continue with next batch instead of failing completely
        
        # Update final status
        progress_bar.progress(0.95)  # Almost done
        status_text.write("Finalizing vector database...")
        
        # Store in session state
        st.session_state.vector_db_instance = vectorstore
        st.session_state.vector_db_type = "langchain_chroma"
        st.session_state.db = vectorstore
        st.session_state.processed_docs = successful_docs > 0
        st.session_state.db_status = f"Healthy (LangChain ChromaDB - {successful_docs} chunks)"
        
        # Store document info in session state for the UI
        st.session_state.documents = []
        for doc in documents:
            if doc.get("content"):
                st.session_state.documents.append({
                    "name": doc.get("name", "unknown"),
                    "path": doc.get("path", ""),
                    "source": doc.get("name", "unknown")
                })
        
        # Log completion time
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Vector database created in {processing_time:.2f} seconds")
        
        # Complete the progress bar and show final status
        progress_bar.progress(1.0)
        status_text.write(f"✅ Completed! Processed {successful_docs} chunks from {len(documents)} documents in {processing_time:.1f} seconds.")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.session_state.db_status = f"Error: Failed to create vector database: {str(e)}"
        return None

def check_db_status():
    """Check vector database status"""
    try:
        # Check SQLite version compatibility
        if not CAN_USE_CHROMA:
            error_msg = f"SQLite version {SQLITE_VERSION} is not compatible with ChromaDB (requires >= {MIN_SQLITE_VERSION})"
            logger.error(error_msg)
            st.warning(f"ChromaDB is not compatible with your SQLite version. {error_msg}")
            st.session_state.db_status = f"Error: {error_msg}"
            st.session_state.processed_docs = False
            return False
        
        # Check if ChromaDB directory exists
        if not os.path.exists(CHROMA_PATH):
            logger.warning(f"ChromaDB directory {CHROMA_PATH} does not exist")
            os.makedirs(CHROMA_PATH, exist_ok=True)
            st.warning("ChromaDB directory created. Please process your documents.")
            st.session_state.db_status = "Needs initialization"
            st.session_state.processed_docs = False
            return False
        
        # Check if we have a valid vector database instance
        if 'vector_db_instance' not in st.session_state or st.session_state.vector_db_instance is None:
            logger.info("Initializing vector database")
            
            # Create embedding function
            if OPENAI_API_KEY:
                embedding_function = CustomOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            else:
                st.warning("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets.")
                st.session_state.db_status = "Error: OpenAI API key is missing"
                st.session_state.processed_docs = False
                return False
            
            try:
                # Initialize ChromaDB with client settings to avoid tenant issues
                from chromadb.config import Settings
                from chromadb import PersistentClient
                
                # Create client with explicit settings
                client = PersistentClient(
                    path=CHROMA_PATH,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
                
                # Create or get collection
                collection_name = "alnu_health"
                try:
                    # Try to get existing collection
                    collection = client.get_collection(collection_name)
                    logger.info(f"Using existing collection: {collection_name}")
                except Exception as e:
                    # Create new collection if it doesn't exist
                    logger.info(f"Creating new collection: {collection_name}")
                    collection = client.create_collection(collection_name)
                
                # Initialize Chroma with the collection
                vectorstore = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embedding_function,
                    persist_directory=CHROMA_PATH
                )
                
                # Store in session state
                st.session_state.vector_db_instance = vectorstore
                st.session_state.vector_db_type = "langchain_chroma"
                st.session_state.db = vectorstore
                
                # Check if there are any documents in the database
                count = collection.count()
                
                if count > 0:
                    st.session_state.processed_docs = True
                    st.session_state.db_status = f"Healthy (LangChain ChromaDB - {count} documents)"
                    logger.info(f"Successfully loaded existing ChromaDB with {count} documents")
                else:
                    st.session_state.processed_docs = False
                    st.session_state.db_status = "Healthy (LangChain ChromaDB - Empty)"
                    logger.info("Successfully initialized empty ChromaDB")
                
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {str(e)}")
                st.warning(f"Failed to initialize vector database: {str(e)}")
                st.session_state.db_status = f"Error: {str(e)}"
                st.session_state.processed_docs = False
                return False
        
        # Update status if not already set
        if 'db_status' not in st.session_state or not st.session_state.db_status:
            # Check if there are any documents in the database
            vectorstore = st.session_state.vector_db_instance
            collection = vectorstore._collection
            count = collection.count()
            
            if count > 0:
                st.session_state.processed_docs = True
                st.session_state.db_status = f"Healthy (LangChain ChromaDB - {count} documents)"
            else:
                st.session_state.processed_docs = False
                st.session_state.db_status = "Healthy (LangChain ChromaDB - Empty)"
        
        return True
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False
