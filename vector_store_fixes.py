import logging
from typing import List, Any, Union
import os

logger = logging.getLogger(__name__)

def safe_content_analysis(documents: List[Any]) -> List[str]:
    """
    Safely extract content from documents, handling different input types.
    
    Args:
        documents: List of documents that might be strings, objects with .content, or other types
        
    Returns:
        List of string content
    """
    analyzed_content = []
    
    for i, doc in enumerate(documents):
        try:
            if isinstance(doc, str):
                # If it's already a string, use it directly
                analyzed_content.append(doc)
            elif hasattr(doc, 'page_content'):
                # LangChain Document object
                analyzed_content.append(doc.page_content)
            elif hasattr(doc, 'content'):
                # Object with content attribute
                analyzed_content.append(doc.content)
            elif hasattr(doc, 'text'):
                # Object with text attribute
                analyzed_content.append(doc.text)
            elif isinstance(doc, dict):
                # Dictionary - try common content keys
                content = (doc.get('content') or 
                          doc.get('text') or 
                          doc.get('page_content') or 
                          str(doc))
                analyzed_content.append(content)
            else:
                # Fallback - convert to string
                logger.warning(f"Document {i} has unexpected type {type(doc)}, converting to string")
                analyzed_content.append(str(doc))
                
        except Exception as e:
            logger.error(f"Error analyzing document {i}: {str(e)}")
            logger.error(f"Document type: {type(doc)}")
            # Add empty string as fallback
            analyzed_content.append("")
    
    return analyzed_content

def fix_vector_store_similarity_search(vector_store, query_embedding: List[float], k: int = 5):
    """
    Fixed similarity search that handles content extraction properly.
    
    Args:
        vector_store: The vector store instance
        query_embedding: The query embedding vector
        k: Number of results to return
        
    Returns:
        List of documents with proper content handling
    """
    try:
        # Perform the similarity search
        results = vector_store.similarity_search(query_embedding, k=k)
        
        # Fix any content issues in the results
        fixed_results = []
        for result in results:
            try:
                if isinstance(result, str):
                    # Create a simple document-like object
                    class SimpleDoc:
                        def __init__(self, content, metadata=None):
                            self.page_content = content
                            self.metadata = metadata or {}
                    
                    fixed_results.append(SimpleDoc(result))
                elif hasattr(result, 'page_content'):
                    # Already a proper document
                    fixed_results.append(result)
                elif hasattr(result, 'content'):
                    # Convert content attribute to page_content
                    class SimpleDoc:
                        def __init__(self, content, metadata=None):
                            self.page_content = content
                            self.metadata = metadata or {}
                    
                    fixed_results.append(SimpleDoc(result.content, getattr(result, 'metadata', {})))
                else:
                    # Create document from string representation
                    class SimpleDoc:
                        def __init__(self, content, metadata=None):
                            self.page_content = content
                            self.metadata = metadata or {}
                    
                    fixed_results.append(SimpleDoc(str(result)))
                    
            except Exception as e:
                logger.error(f"Error fixing result: {str(e)}")
                # Skip problematic results
                continue
        
        return fixed_results
        
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        return []

def create_document_wrapper(content: str, metadata: dict = None):
    """Create a simple document wrapper that's compatible with LangChain."""
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    return Document(content, metadata)

class SafeEmbeddingsWrapper:
    """Wrapper for embeddings that safely handles different input types."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def embed_query(self, text):
        """Safely embed a query, handling different input types."""
        try:
            # Handle case where text might be a list
            if isinstance(text, list):
                if len(text) == 0:
                    logger.warning("Empty list provided for embedding")
                    return [0.0] * getattr(self.embeddings, 'output_dim', 768)
                # If it's a list of numbers, assume it's already an embedding
                if all(isinstance(x, (int, float)) for x in text):
                    return list(text)
                # If it's a list of strings, join them
                text = " ".join(str(item) for item in text)
            
            # Convert to string if needed
            if not isinstance(text, str):
                text = str(text)
            
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error in safe embed_query: {str(e)}")
            # Return a fallback embedding
            return [0.0] * getattr(self.embeddings, 'output_dim', 768)
    
    def embed_documents(self, texts):
        """Safely embed documents."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error in safe embed_documents: {str(e)}")
            # Return fallback embeddings
            return [[0.0] * getattr(self.embeddings, 'output_dim', 768) for _ in texts]
    
    def __getattr__(self, name):
        """Pass through other attributes to the wrapped embeddings."""
        return getattr(self.embeddings, name) 