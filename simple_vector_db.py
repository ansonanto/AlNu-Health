"""
Simple FAISS-based vector database implementation for AlNu Health.
This provides a more compatible alternative to ChromaDB for Streamlit deployment.
"""
import os
import time
import pickle
import logging
import numpy as np
import faiss
import streamlit as st
from typing import List, Dict, Any, Optional

# OpenAI for embeddings
from openai import OpenAI

# Local imports
from config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
STORAGE_DIR = "./simple_vector_storage"
FAISS_INDEX_FILE = os.path.join(STORAGE_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(STORAGE_DIR, "metadata.pkl")

class SimpleVectorDB:
    """Simple FAISS-based vector database implementation"""
    
    def __init__(self):
        """Initialize the vector database"""
        self.client = None
        self.index = None
        self.metadata = []
        self.embedding_dim = 1536  # OpenAI embedding dimension
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            logger.error("OpenAI API key is missing")
            raise ValueError("OpenAI API key is required for embeddings")
    
    def create_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        return self.index
    
    def load(self):
        """Load the index and metadata from disk"""
        try:
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
                # Load the FAISS index
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                
                # Load the metadata
                with open(METADATA_FILE, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
                return True
            else:
                logger.info("No existing index found, creating a new one")
                self.create_index()
                return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            self.create_index()
            return False
    
    def save(self):
        """Save the index and metadata to disk"""
        try:
            # Create the directory if it doesn't exist
            os.makedirs(STORAGE_DIR, exist_ok=True)
            
            # Save the FAISS index
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            
            # Save the metadata
            with open(METADATA_FILE, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def chunk_text(self, text, max_chunk_size=4000):
        """Split text into chunks to avoid token limits"""
        # Simple chunking by characters
        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunks.append(text[i:i+max_chunk_size])
        return chunks
    
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts with chunking for large documents"""
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # First, chunk any large texts
            chunked_texts = []
            text_to_chunks_map = {}  # Maps original text index to chunk indices
            
            for i, text in enumerate(texts):
                # Check if text is too large (approximate token count)
                if len(text) > 6000:  # Conservative estimate (about 1500 tokens)
                    chunks = self.chunk_text(text)
                    chunk_start_idx = len(chunked_texts)
                    chunked_texts.extend(chunks)
                    text_to_chunks_map[i] = list(range(chunk_start_idx, chunk_start_idx + len(chunks)))
                    logger.info(f"Split document {i} into {len(chunks)} chunks")
                else:
                    chunked_texts.append(text)
                    text_to_chunks_map[i] = [len(chunked_texts) - 1]
            
            # Now process the chunked texts in batches
            all_embeddings = []
            batch_size = 10
            for i in range(0, len(chunked_texts), batch_size):
                batch = chunked_texts[i:i+batch_size]
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Embedded batch {i//batch_size + 1}/{(len(chunked_texts) + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                    raise
            
            # Now combine the embeddings for chunks of the same original text
            final_embeddings = []
            for i in range(len(texts)):
                chunk_indices = text_to_chunks_map[i]
                if len(chunk_indices) == 1:
                    # If text wasn't chunked, just use its embedding
                    final_embeddings.append(all_embeddings[chunk_indices[0]])
                else:
                    # If text was chunked, average the embeddings of its chunks
                    chunk_embeddings = [all_embeddings[idx] for idx in chunk_indices]
                    avg_embedding = [sum(x) / len(chunk_embeddings) for x in zip(*chunk_embeddings)]
                    final_embeddings.append(avg_embedding)
                    logger.info(f"Averaged embeddings for document {i} from {len(chunk_indices)} chunks")
            
            return final_embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def add_documents(self, documents):
        """Add documents to the index with chunking for large documents"""
        try:
            if not self.index:
                self.create_index()
            
            # Process documents in smaller batches to avoid memory issues
            batch_size = 5  # Process 5 documents at a time
            total_docs = len(documents)
            successful_docs = 0
            
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                batch = documents[batch_start:batch_end]
                
                try:
                    # Extract text and metadata for this batch
                    texts = []
                    valid_indices = []
                    
                    for i, doc in enumerate(batch):
                        if doc.get("content"):
                            texts.append(doc["content"])
                            valid_indices.append(i)
                    
                    if not texts:
                        logger.warning(f"No valid document content in batch {batch_start//batch_size + 1}")
                        continue
                    
                    # Get embeddings for this batch
                    logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} with {len(texts)} documents")
                    embeddings = self.get_embeddings(texts)
                    
                    # Convert to numpy array
                    embeddings_np = np.array(embeddings).astype('float32')
                    
                    # Add to index
                    self.index.add(embeddings_np)
                    
                    # Add metadata for valid documents
                    for i, idx in enumerate(valid_indices):
                        doc = batch[idx]
                        self.metadata.append({
                            "name": doc.get("name", "unknown"),
                            "path": doc.get("path", ""),
                            "source": doc.get("name", "unknown"),
                            "content": doc.get("content", "")
                        })
                    
                    successful_docs += len(texts)
                    logger.info(f"Successfully processed batch {batch_start//batch_size + 1}")
                    
                    # Save after each batch to avoid losing progress
                    self.save()
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_start//batch_size + 1}: {str(e)}")
                    # Continue with next batch instead of failing completely
            
            logger.info(f"Added {successful_docs}/{total_docs} documents to index")
            return successful_docs > 0
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(self, query, k=5):
        """Search for similar documents"""
        try:
            if not self.index or self.index.ntotal == 0:
                logger.warning("Index is empty or not initialized")
                return []
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_embedding_np, min(k, self.index.ntotal))
            
            # Get results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS returns -1 for not enough results
                    results.append({
                        "content": self.metadata[idx]["content"],
                        "metadata": {
                            "source": self.metadata[idx]["name"],
                            "score": float(1.0 - distances[0][i] / 100.0)  # Convert distance to score
                        }
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def similarity_search_with_relevance_scores(self, query, k=5):
        """Search for similar documents with relevance scores"""
        results = self.similarity_search(query, k)
        return [(result, result["metadata"]["score"]) for result in results]

# Initialize vector database
def initialize_vector_db(reset_db=False) -> Optional[Any]:
    """Initialize vector database"""
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
        
        # Create vector database instance
        db = SimpleVectorDB()
        
        # Reset if requested
        if reset_db and os.path.exists(FAISS_INDEX_FILE):
            logger.info(f"Resetting vector database at {FAISS_INDEX_FILE}")
            try:
                os.remove(FAISS_INDEX_FILE)
                os.remove(METADATA_FILE)
                logger.info("Vector database reset successfully")
            except Exception as e:
                logger.error(f"Error resetting vector database: {str(e)}")
        
        # Load or create new index
        db.load()
        
        # Store in session state
        st.session_state.vector_db_instance = db
        st.session_state.vector_db_type = "simple_faiss"
        st.session_state.db = db
        st.session_state.processed_docs = db.index.ntotal > 0 if db.index else False
        st.session_state.db_status = f"Healthy (Simple FAISS - {db.index.ntotal if db.index else 0} documents)"
        
        logger.info("Successfully initialized Simple FAISS vector database")
        return db
        
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
        
        # Initialize or get existing database
        if update_existing and 'vector_db_instance' in st.session_state and st.session_state.vector_db_instance is not None:
            db = st.session_state.vector_db_instance
        else:
            db = initialize_vector_db(reset_db=not update_existing)
        
        if not db:
            logger.error("Failed to initialize vector database")
            return None
        
        # Add documents
        success = db.add_documents(documents)
        
        if success:
            # Update session state
            st.session_state.vector_db_instance = db
            st.session_state.vector_db_type = "simple_faiss"
            st.session_state.db = db
            st.session_state.processed_docs = True
            st.session_state.db_status = f"Healthy (Simple FAISS - {db.index.ntotal if db.index else 0} documents)"
            
            # Log completion time
            end_time = time.time()
            logger.info(f"Vector database created in {end_time - start_time:.2f} seconds")
            
            return db
        else:
            logger.error("Failed to add documents to vector database")
            return None
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
        
        # Get database instance
        db = st.session_state.vector_db_instance
        
        # Update status
        if db and hasattr(db, 'index') and db.index:
            st.session_state.db_status = f"Healthy (Simple FAISS - {db.index.ntotal} documents)"
            st.session_state.processed_docs = db.index.ntotal > 0
        else:
            st.session_state.db_status = "Ready (Simple FAISS - Empty)"
            st.session_state.processed_docs = False
        
        return True
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        st.session_state.db_status = f"Error: {str(e)}"
        return False
