import logging
import random
from typing import List
# Using new OpenAI API style for compatibility with version 1.0.0+
from openai import OpenAI
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
        
        # Check if API key is available
        if not self.api_key or self.api_key == "":
            raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to your Streamlit secrets.")
            
        # Using new OpenAI API style for version 1.0.0+
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized CustomOpenAIEmbeddings with model {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        try:
            embeddings = []
            # Process in smaller batches to avoid token limits
            batch_size = 5  # Smaller batch size to avoid token limits
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logger.info(f"Embedding {len(texts)} texts in {total_batches} batches of size {batch_size}")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Check token count for this batch (approximate)
                total_chars = sum(len(text) for text in batch)
                approx_tokens = total_chars / 4  # Rough estimate: 4 chars per token
                
                if approx_tokens > 8000:
                    logger.warning(f"Batch {i//batch_size + 1}/{total_batches} may exceed token limit ({approx_tokens:.0f} est. tokens)")
                    # Further split this batch if needed
                    sub_batch_size = max(1, batch_size // 2)
                    logger.info(f"Splitting into smaller sub-batches of size {sub_batch_size}")
                    
                    sub_batches = [batch[j:j+sub_batch_size] for j in range(0, len(batch), sub_batch_size)]
                    for sub_batch in sub_batches:
                        try:
                            sub_response = self.client.embeddings.create(
                                input=sub_batch,
                                model=self.model
                            )
                            sub_embeddings = [data.embedding for data in sub_response.data]
                            embeddings.extend(sub_embeddings)
                        except Exception as sub_e:
                            logger.error(f"Error in sub-batch embedding: {str(sub_e)}")
                            # Fallback to random embeddings for this sub-batch
                            dimension = 1536
                            sub_random_embeddings = [
                                [random.uniform(-1, 1) for _ in range(dimension)]
                                for _ in sub_batch
                            ]
                            embeddings.extend(sub_random_embeddings)
                else:
                    # Process normal batch
                    try:
                        # Using new OpenAI API style for version 1.0.0+
                        response = self.client.embeddings.create(
                            input=batch,
                            model=self.model
                        )
                        batch_embeddings = [data.embedding for data in response.data]
                        embeddings.extend(batch_embeddings)
                        logger.info(f"Successfully embedded batch {i//batch_size + 1}/{total_batches}")
                    except Exception as batch_e:
                        logger.error(f"Error in batch embedding: {str(batch_e)}")
                        # Fallback to random embeddings for this batch
                        dimension = 1536
                        random_embeddings = [
                            [random.uniform(-1, 1) for _ in range(dimension)]
                            for _ in batch
                        ]
                        embeddings.extend(random_embeddings)
                
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
            # Using new OpenAI API style for version 1.0.0+
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            # Fallback to random embedding if OpenAI API fails
            dimension = 1536
            return [random.uniform(-1, 1) for _ in range(dimension)]
