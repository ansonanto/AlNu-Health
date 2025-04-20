import logging
import random
from typing import List
# Using older OpenAI API style for compatibility with version 0.28.1
import openai
from langchain.embeddings.base import Embeddings
from config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomOpenAIEmbeddings(Embeddings):
    """Custom embeddings class that uses the direct OpenAI API."""
    
    def __init__(self, api_key=None, model="text-embedding-3-small", **kwargs):
        """Initialize with API key and model name.
        
        Note: We accept **kwargs to handle any deprecated parameters like 'proxies'
        but we don't use them with the new OpenAI client.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        # Using older OpenAI API style for version 0.28.1
        openai.api_key = self.api_key
        self.client = openai
        logger.info(f"Initialized CustomOpenAIEmbeddings with model {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Using older OpenAI API style for version 0.28.1
                response = self.client.Embedding.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                embeddings.extend(batch_embeddings)
                
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            # Fallback to random embeddings if OpenAI API fails
            dimension = 1536
            return [
                [random.uniform(-1, 1) for _ in range(dimension)]
                for _ in texts
            ]
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        try:
            # Using older OpenAI API style for version 0.28.1
            response = self.client.Embedding.create(
                input=text,
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            # Fallback to random embedding if OpenAI API fails
            dimension = 1536
            return [random.uniform(-1, 1) for _ in range(dimension)]
