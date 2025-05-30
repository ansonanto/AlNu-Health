"""
LangChain Google Gemini Embeddings implementation with batch processing support.
This replaces the custom GeminiEmbeddings implementation with the official LangChain integration.
"""

import logging
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY, EMBEDDING_DIMENSION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainGeminiEmbeddings:
    """Wrapper for LangChain's GoogleGenerativeAIEmbeddings to maintain compatibility with existing code."""
    
    def __init__(self, api_key=None, output_dim=None, batch_size=10, **kwargs):
        """Initialize with API key.
        
        Args:
            api_key: The API key for Gemini API
            output_dim: The dimension of the embeddings (default is from config.EMBEDDING_DIMENSION)
            batch_size: Number of documents to process in a batch
            **kwargs: Additional parameters passed to GoogleGenerativeAIEmbeddings
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.output_dim = output_dim if output_dim is not None else EMBEDDING_DIMENSION
        self.batch_size = batch_size
        
        # Initialize the LangChain GoogleGenerativeAIEmbeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model="models/embedding-001",  # Default Gemini embedding model
            task_type="RETRIEVAL_DOCUMENT",  # Default task type for documents
            **kwargs
        )
        
        logger.info(f"Initialized LangChainGeminiEmbeddings with batch size {self.batch_size}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents with batch processing."""
        try:
            # The LangChain implementation already handles batching internally
            embeddings = self.embeddings.embed_documents(texts)
            
            # Check if the embedding dimension matches our expected dimension
            if embeddings and len(embeddings[0]) != self.output_dim:
                logger.warning(f"Embedding dimension mismatch: got {len(embeddings[0])}, expected {self.output_dim}")
                # Update our dimension to match what we received
                self.output_dim = len(embeddings[0])
                logger.info(f"Adapting to actual embedding dimension: {self.output_dim}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            # Return empty list on error
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        try:
            # Override the task type for queries
            query_embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=self.api_key,
                model="models/embedding-001",
                task_type="RETRIEVAL_QUERY"
            )
            return query_embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            # Return empty list on error
            return []
