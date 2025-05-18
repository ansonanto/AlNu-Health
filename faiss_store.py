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

        # Process chunks in batches for embeddings
        try:
            batch_size = 20  # Process 20 chunks at a time
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
                
                # Get embeddings for the batch
                batch_embeddings = self.embedding_function.embed_documents(batch_chunks)
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array and add to FAISS index
            embeddings_np = np.array(all_embeddings).astype('float32')
            self.index.add(embeddings_np)
            
            # Store metadata
            self.metadata.extend(all_metadatas)
            
            # Save to disk
            self._save_index()
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.write("✅ Processing complete!")
            
            logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents to vector store")
                    
            # Save to disk after all batches are processed
            self._save_index()
            logger.info(f"Added {len(documents)} documents to vector store")
            
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
        from config import OPENAI_API_KEY
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
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
