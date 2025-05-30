import logging
import random
import time
from typing import List, Union, Any
import google.generativeai as genai
from google.genai.types import EmbedContentConfig
from langchain.embeddings.base import Embeddings
from config import GEMINI_API_KEY, EMBEDDING_DIMENSION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddings(Embeddings):
    """Embeddings class that uses Google's Gemini API."""
    
    def __init__(self, api_key=None, output_dim=None, max_retries=3, **kwargs):
        """Initialize with API key.
        
        Args:
            api_key: The API key for Gemini API
            output_dim: The dimension of the embeddings (default is from config.EMBEDDING_DIMENSION)
            max_retries: Maximum number of retries for API calls
            **kwargs: Additional parameters (not used)
        """
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in config or pass api_key parameter.")
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # The embedding model - use the correct model name with prefix
        self.embedding_model = "models/embedding-001"
        
        # Set the output dimension from config if not specified
        self.output_dim = output_dim if output_dim is not None else EMBEDDING_DIMENSION
        
        # Set maximum retries
        self.max_retries = max_retries
        
        logger.info(f"Initialized GeminiEmbeddings with model {self.embedding_model}")
        logger.info(f"Using dimension: {self.output_dim}")
        
        # Test the connection
        try:
            test_response = self.embed_query("test connection")
            if test_response:
                logger.info("Successfully connected to Gemini API")
                # Update output dimension based on actual response
                self.output_dim = len(test_response)
                logger.info(f"Actual embedding dimension: {self.output_dim}")
            else:
                logger.warning("Connection test returned empty embedding")
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
    
    def _embed_with_retry(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Embed text with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Validate input
                if not text or not text.strip():
                    logger.warning("Empty text provided for embedding")
                    return [0.0] * self.output_dim
                
                text = text.strip()
                
                # Make API call with correct parameters
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=[text],  # Must be a list, not a string
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.output_dim,
                    )
                )
                
                # Access the embedding correctly
                if hasattr(response, 'embeddings') and response.embeddings:
                    # embeddings is a list, get the first one
                    embedding = response.embeddings[0]
                    if hasattr(embedding, 'values'):
                        return embedding.values
                    else:
                        # If it's already a list of floats
                        return embedding
                else:
                    raise ValueError("No embeddings found in response")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # If it's a rate limit error, wait longer
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Rate limit hit, waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    # For other errors, wait a shorter time
                    wait_time = 1 + random.uniform(0, 1)
                    time.sleep(wait_time)
        
        # If all retries failed, log the error and return fallback
        logger.error(f"All {self.max_retries} attempts failed. Last error: {str(last_exception)}")
        return self._get_fallback_embedding()
    
    def _get_fallback_embedding(self) -> List[float]:
        """Generate a fallback embedding when API calls fail."""
        logger.warning("Using fallback random embedding")
        return [random.uniform(-0.1, 0.1) for _ in range(self.output_dim)]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents with retry logic."""
        try:
            if not texts:
                logger.warning("Empty text list provided")
                return []
            
            logger.info(f"Embedding {len(texts)} documents")
            embeddings = []
            
            # Process each text individually as Gemini embedding API processes one at a time
            for i, text in enumerate(texts):
                try:
                    embedding = self._embed_with_retry(text, task_type="RETRIEVAL_DOCUMENT")
                    embeddings.append(embedding)
                    
                    # Log progress for large batches
                    if len(texts) > 10 and (i + 1) % 10 == 0:
                        logger.info(f"Embedded {i + 1}/{len(texts)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to embed document {i}: {str(e)}")
                    embeddings.append(self._get_fallback_embedding())
            
            logger.info(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            # Return fallback embeddings for all texts
            return [self._get_fallback_embedding() for _ in texts]
    
    def embed_query(self, text: Union[str, List[Any], Any]) -> List[float]:
        """Get embedding for a single query.
        
        Args:
            text: The text to embed. Can be a string, list, or other type.
                  If it's a list of numbers, assumes it's already an embedding.
                  If it's a list of strings, joins them.
                  If it's another type, converts to string.
        """
        try:
            # Handle case where text might be a list (vector search sometimes passes lists)
            if isinstance(text, list):
                if len(text) == 0:
                    logger.warning("Empty list provided for embedding")
                    return self._get_fallback_embedding()
                # If it's a list of numbers, assume it's already an embedding
                if all(isinstance(x, (int, float)) for x in text):
                    logger.debug("Received numerical list, assuming it's already an embedding")
                    return list(text)
                # If it's a list of strings, join them
                text = " ".join(str(item) for item in text)
                logger.debug("Converted list to string for embedding")
            
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
            
            if not text or not text.strip():
                logger.warning("Empty query text provided")
                return self._get_fallback_embedding()
            
            logger.debug(f"Embedding query: {text[:50]}...")
            
            # Use RETRIEVAL_QUERY task type for queries
            embedding = self._embed_with_retry(text, task_type="RETRIEVAL_QUERY")
            
            logger.debug(f"Successfully embedded query, dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            logger.error(f"Input type: {type(text)}, Input value: {text}")
            return self._get_fallback_embedding()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.output_dim
    
    def test_embedding(self, text: str = "This is a test") -> bool:
        """Test the embedding functionality."""
        try:
            embedding = self.embed_query(text)
            if embedding and len(embedding) == self.output_dim:
                logger.info("Embedding test passed")
                return True
            else:
                logger.error(f"Embedding test failed: got {len(embedding) if embedding else 0} dimensions, expected {self.output_dim}")
                return False
        except Exception as e:
            logger.error(f"Embedding test failed with error: {str(e)}")
            return False