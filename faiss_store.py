import os
import json
import faiss
import pickle
import logging
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.retriever import BaseRetriever
from langchain.schema.embeddings import Embeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_function: Embeddings, index_folder: str = "simple_vector_storage"):
        self.embedding_function = embedding_function
        self.index_folder = index_folder
        
        # Use consistent file names for FAISS index files
        self.index_path = os.path.join(index_folder, "index.faiss")
        self.metadata_path = os.path.join(index_folder, "index.pkl")
        
        # Import the correct dimension from config
        from config import EMBEDDING_DIMENSION
        self.dimension = EMBEDDING_DIMENSION
        logger.info(f"Using embedding dimension from config: {self.dimension}")
        
        # Store dimension in a separate file for persistence
        self.dimension_path = os.path.join(index_folder, "dimension.txt")
        
        # Create storage directory if it doesn't exist
        os.makedirs(index_folder, exist_ok=True)
        
        # Initialize or load the index
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()

    def _save_dimension(self):
        """Save the dimension information to a file for persistence"""
        try:
            with open(self.dimension_path, 'w') as f:
                f.write(str(self.dimension))
            logger.info(f"Saved dimension information: {self.dimension}")
        except Exception as e:
            logger.error(f"Error saving dimension information: {str(e)}")
    
    def _load_dimension(self) -> int:
        """Load the dimension information from file if it exists"""
        if os.path.exists(self.dimension_path):
            try:
                with open(self.dimension_path, 'r') as f:
                    dimension = int(f.read().strip())
                logger.info(f"Loaded dimension from file: {dimension}")
                return dimension
            except Exception as e:
                logger.error(f"Error loading dimension from file: {str(e)}")
        return self.dimension  # Return default if file doesn't exist or error occurs
    
    def _load_or_create_index(self) -> faiss.Index:
        """Load existing index or create a new one"""
        # First try to load the dimension from file for persistence
        saved_dimension = self._load_dimension()
        if saved_dimension != self.dimension:
            logger.info(f"Using saved dimension {saved_dimension} instead of config dimension {self.dimension}")
            self.dimension = saved_dimension
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                index = faiss.read_index(self.index_path)
                
                # Check if the index dimension matches our expected dimension
                index_dimension = index.d
                if index_dimension != self.dimension:
                    logger.warning(f"Dimension mismatch: index has {index_dimension}, saved has {self.dimension}")
                    # We'll handle this by using the index's dimension instead of the saved dimension
                    self.dimension = index_dimension
                    logger.info(f"Adapting to existing index dimension: {self.dimension}")
                    # Save the updated dimension for future consistency
                    self._save_dimension()
                
                return index
            except Exception as e:
                logger.error(f"Error loading existing index: {str(e)}")
        
        # Create a new index if we couldn't load an existing one
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        index = faiss.IndexFlatL2(self.dimension)
        self._save_dimension()
        return index

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load existing metadata or initialize empty list"""
        if os.path.exists(self.metadata_path):
            logger.info(f"Loading existing metadata from {self.metadata_path}")
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
        
        return []

    def _save_index(self):
        """Save the FAISS index to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        # Also save the dimension information for persistence
        self._save_dimension()
        logger.info(f"Saved FAISS index and metadata with dimension {self.dimension}")

    def add_documents(self, documents: List[Document], update_existing: bool = False):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return

        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Process each document
        all_chunks = []
        all_metadatas = []
        
        # Create progress bar for document processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, doc in enumerate(documents):
            # Update progress
            progress = (i + 1) / len(documents)
            progress_bar.progress(progress)
            status_text.write(f"Processing document {i + 1}/{len(documents)}")
            
            # Skip empty documents
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(f"Skipping empty document: {doc.metadata.get('source', 'Unknown')}")
                continue
                
            try:
                # Split text into chunks
                chunks = text_splitter.split_text(doc.page_content)
                
                # Get or create metadata
                doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Ensure source and title are present
                if 'source' not in doc_metadata:
                    doc_metadata['source'] = f'Document_{i+1}'
                if 'title' not in doc_metadata:
                    doc_metadata['title'] = os.path.basename(doc_metadata['source'])
                    
                # Create chunk-specific metadata
                for j, chunk in enumerate(chunks):
                    # Skip empty chunks
                    if not chunk or not chunk.strip():
                        continue
                        
                    chunk_metadata = {
                        'content': chunk,
                        'metadata': {
                            **doc_metadata,
                            'chunk': j + 1,
                            'total_chunks': len(chunks)
                        }
                    }
                    all_metadatas.append(chunk_metadata)
                    all_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {str(e)}")
                # Continue with next document
                continue

        # Process chunks in batches for embeddings
        try:
            if not all_chunks:
                logger.warning("No valid chunks to process")
                progress_bar.progress(1.0)
                status_text.write("⚠️ No valid content to process")
                return
                
            batch_size = 5  # Reduced from 20 to 5 to avoid timeouts
            all_embeddings = []
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            # Process each batch
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                current_batch = i // batch_size + 1
                
                # Update progress
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.write(f"Generating embeddings: Batch {current_batch}/{total_batches}")
                
                try:
                    # Get embeddings for the batch
                    batch_embeddings = self.embedding_function.embed_documents(batch_chunks)
                    
                    # Verify all embeddings have the same dimension
                    if batch_embeddings and len(batch_embeddings) > 0:
                        first_dim = len(batch_embeddings[0])
                        if first_dim != self.dimension:
                            logger.warning(f"Embedding dimension mismatch: got {first_dim}, expected {self.dimension}")
                            # Update the dimension if it's different
                            if i == 0:  # Only update on first batch
                                logger.info(f"Updating FAISS index dimension from {self.dimension} to {first_dim}")
                                self.dimension = first_dim
                                self.index = faiss.IndexFlatL2(self.dimension)
                    
                    all_embeddings.extend(batch_embeddings)
                    
                    # Add a small delay between batches to avoid rate limits
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {current_batch}: {str(e)}")
                    # Skip this batch and continue with the next one
                    continue
            
            # Check if we have any valid embeddings
            if not all_embeddings:
                logger.error("No valid embeddings generated")
                progress_bar.progress(1.0)
                status_text.write("❌ Failed to generate embeddings")
                return
                
            # Convert to numpy array and add to FAISS index
            try:
                embeddings_np = np.array(all_embeddings).astype('float32')
                self.index.add(embeddings_np)
                
                # Store metadata
                self.metadata.extend(all_metadatas[:len(all_embeddings)])  # Only store metadata for successful embeddings
                
                # Save to disk
                self._save_index()
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.write("✅ Processing complete!")
                
                logger.info(f"Added {len(all_embeddings)} chunks from {len(documents)} documents to vector store")
            except Exception as e:
                logger.error(f"Error adding embeddings to FAISS index: {str(e)}")
                progress_bar.progress(1.0)
                status_text.write("❌ Error adding embeddings to index")
                raise
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            progress_bar.progress(1.0)
            status_text.write(f"❌ Error: {str(e)}")
            raise

    def get(self) -> Dict[str, Any]:
        """Get all documents and metadata"""
        metadatas = []
        documents = []
        for item in self.metadata:
            if "metadata" in item and "content" in item:
                metadatas.append(item["metadata"])
                documents.append(item["content"])
        return {
            "metadatas": metadatas,
            "documents": documents
        }

    def similarity_search(self, query: str, k: int = 4, filter_references: bool = True, buffer_size: int = 5) -> List[Document]:
        """Return documents most similar to query.
        
        Args:
            query: The search query
            k: Number of documents to return
            filter_references: Whether to filter out reference sections
            buffer_size: Extra documents to retrieve as buffer for filtering
        """
        # Get more documents than needed if filtering is enabled
        search_k = k + buffer_size if filter_references else k
        
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Convert to numpy array with correct shape
        xq = np.array([query_embedding], dtype=np.float32)
        
        # Search the index
        D, I = self.index.search(xq, search_k)
        
        # Get documents from the index
        docs = []
        for idx in I[0]:
            if idx != -1:  # FAISS returns -1 for empty slots
                metadata = self.metadata[idx]
                doc = Document(
                    page_content=metadata['content'],
                    metadata=metadata['metadata']
                )
                docs.append(doc)
        
        if filter_references:
            # Filter out reference sections and get top k
            content_filter = ContentQualityFilter()
            docs = content_filter.filter_documents(docs, k)
        else:
            # Just take top k if no filtering
            docs = docs[:k]
        
        return docs

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, filter_references: bool = True, buffer_size: int = 5
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query along with relevance scores."""
        # Get embedding for query
        query_embedding = self.embedding_function.embed_query(query)
        
        # Search the FAISS index with buffer for filtering
        search_k = k * buffer_size if filter_references else k
        D, I = self.index.search(np.array([query_embedding]), search_k)
        
        # Get documents and scores
        docs = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx == -1:  # This means no more results from FAISS
                continue
            doc = Document(
                page_content=self.metadata[idx]["content"],
                metadata=self.metadata[idx]["metadata"]
            )
            # Convert distance to similarity score
            similarity = 1.0 - min(1.0, float(dist))
            docs.append((doc, similarity))
            
        # Filter out reference sections if requested
        if filter_references:
            content_filter = ContentQualityFilter()
            filtered_docs = content_filter.filter_documents([doc for doc, _ in docs], k)
            # Match filtered docs back with their scores
            filtered_docs_with_scores = []
            for filtered_doc in filtered_docs:
                for doc, score in docs:
                    if doc.page_content == filtered_doc.page_content:
                        filtered_docs_with_scores.append((filtered_doc, score))
                        break
            docs = filtered_docs_with_scores[:k]
        else:
            docs = docs[:k]
        
        return docs

    def as_retriever(self, **kwargs) -> BaseRetriever:
        """Create a retriever from this vector store"""
        return FAISSRetriever(vectorstore=self, **kwargs)


class ContentQualityFilter:
    """Filter document chunks based on content quality."""
    
    def __init__(self):
        from config import GEMINI_CONFIG
        from gemini_llm import GeminiLLM
        self.llm = GeminiLLM(temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content quality analyzer. Your task is to determine if a given text chunk contains meaningful content or is just references/citations. A chunk is considered a reference section if it primarily consists of bibliographic citations (e.g., author names, journal names, DOIs, publication years, [CrossRef], [PubMed] tags) or if it's a list of numbered references. A chunk is considered content if it contains actual explanatory text about the topic."),
            ("user", "Analyze this text chunk and respond with ONLY 'reference' or 'content'. Respond with 'reference' if the chunk is primarily citations or bibliography.\n\n{chunk}")
        ])
    
    def is_reference_section(self, chunk: str) -> bool:
        """Determine if a chunk is just references."""
        try:
            result = self.llm(self.prompt.format_messages(chunk=chunk))
            return result.content.strip().lower() == "reference"
        except Exception as e:
            logger.warning(f"Error in content analysis: {e}")
            return False
    
    def filter_documents(self, docs: List[Document], k: int) -> List[Document]:
        """Filter out reference sections and return top k documents."""
        filtered_docs = []
        
        for doc in docs:
            if len(filtered_docs) >= k:
                break
            
            if not self.is_reference_section(doc.page_content):
                filtered_docs.append(doc)
        
        return filtered_docs[:k]


class FAISSRetriever(BaseRetriever):
    def __init__(self, vectorstore: FAISSVectorStore, k: int = 4, filter_references: bool = True, buffer_size: int = 5):
        self.vectorstore = vectorstore
        self.k = k
        self.filter_references = filter_references
        self.buffer_size = buffer_size

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query"""
        docs = self.vectorstore.similarity_search(
            query, 
            k=self.k, 
            filter_references=self.filter_references,
            buffer_size=self.buffer_size
        )
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query async"""
        raise NotImplementedError("Async retrieval not implemented yet")
