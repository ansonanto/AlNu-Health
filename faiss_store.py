import os
import json
import faiss
import pickle
import logging
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.retriever import BaseRetriever
from langchain.schema.embeddings import Embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, embedding_function: Embeddings, index_folder: str = "simple_vector_storage"):
        self.embedding_function = embedding_function
        self.index_folder = index_folder
        self.index_path = os.path.join(index_folder, "faiss_index.bin")
        self.metadata_path = os.path.join(index_folder, "metadata.pkl")
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Create storage directory if it doesn't exist
        os.makedirs(index_folder, exist_ok=True)
        
        # Initialize or load the index
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()

    def _load_or_create_index(self) -> faiss.Index:
        """Load existing index or create a new one"""
        if os.path.exists(self.index_path):
            logger.info("Loading existing FAISS index")
            return faiss.read_index(self.index_path)
        logger.info("Creating new FAISS index")
        return faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load existing metadata or initialize empty list"""
        if os.path.exists(self.metadata_path):
            logger.info("Loading existing metadata")
            with open(self.metadata_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_index(self):
        """Save the FAISS index to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved FAISS index and metadata")

    def add_documents(self, documents: List[Document], update_existing: bool = False):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return

        if update_existing and os.path.exists(self.index_path):
            logger.info("Resetting existing index")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

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
        for doc in documents:
            # Split text into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Create metadata for each chunk
            doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            chunk_metadatas = [{
                **doc_metadata,
                'chunk': i,
                'total_chunks': len(chunks)
            } for i in range(len(chunks))]
            
            all_chunks.extend(chunks)
            all_metadatas.extend(chunk_metadatas)

        # Process chunks in batches
        try:
            batch_size = 20  # Process 20 chunks at a time
            all_embeddings = []
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                current_batch = i // batch_size + 1
                
                # Update progress
                progress = current_batch / total_batches
                progress_bar.progress(progress)
                status_text.write(f"Generating embeddings: Batch {current_batch}/{total_batches}")
                
                # Process batch
                batch_embeddings = self.embedding_function.embed_documents(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Processed batch {current_batch}/{total_batches}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.write("✅ Embeddings generation complete!")
            
            # Convert to numpy array
            embeddings_np = np.array(all_embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_np)
            
            # Store metadata
            for chunk, metadata in zip(all_chunks, all_metadatas):
                self.metadata.append({
                    "content": chunk,
                    "metadata": metadata
                })
            
            # Save to disk
            self._save_index()
            logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
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

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        # Convert to documents
        documents = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):  # -1 indicates no match
                item = self.metadata[idx]
                doc = Document(
                    page_content=item["content"],
                    metadata=item["metadata"]
                )
                documents.append(doc)
        
        return documents

    def as_retriever(self, **kwargs) -> BaseRetriever:
        """Create a retriever from this vector store"""
        return FAISSRetriever(vectorstore=self, **kwargs)


class FAISSRetriever(BaseRetriever):
    def __init__(self, vectorstore: FAISSVectorStore, k: int = 4):
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query"""
        return self.vectorstore.similarity_search(query, k=self.k)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query async"""
        return self.get_relevant_documents(query)
