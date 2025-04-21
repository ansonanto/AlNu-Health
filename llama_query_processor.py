"""
Query processor for LlamaIndex-based vector database implementation.
"""
import time
import logging
import streamlit as st
from typing import List, Dict, Any, Optional

from openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import OpenAI as LlamaOpenAI

from config import OPENAI_API_KEY, MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_documents(query, db, conversation_history=None, is_summary_request=False):
    """Process query and generate response using RAG approach with LlamaIndex"""
    try:
        start_time = time.time()
        
        # Check if we have a valid database
        if db is None:
            return {
                "response": "Please process documents first in the Document Management tab.",
                "sources": [],
                "chunks": [],
                "processing_time": 0
            }
        
        # Determine number of results to retrieve
        top_k = 3 if is_summary_request else 5
        
        # Create query engine with appropriate parameters
        query_engine = db.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact" if is_summary_request else "tree_summarize",
            llm=LlamaOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
        )
        
        # Execute query
        logger.info(f"Executing query: {query}")
        response = query_engine.query(query)
        
        # Extract source nodes and their metadata
        source_nodes = response.source_nodes
        sources = []
        chunks = []
        
        for node in source_nodes:
            source_info = {
                "source": node.metadata.get("name", "Unknown"),
                "score": node.score if hasattr(node, "score") else None,
                "chunk": node.metadata.get("chunk", 0),
                "total_chunks": node.metadata.get("total_chunks", 1)
            }
            sources.append(source_info)
            
            chunk_info = {
                "content": node.text,
                "metadata": node.metadata
            }
            chunks.append(chunk_info)
            
        # Log the sources for debugging
        logger.info(f"Retrieved {len(sources)} sources:")
        for i, source in enumerate(sources):
            logger.info(f"  Source {i+1}: {source['source']}, score={source['score']}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        # Return the response and metadata
        return {
            "response": str(response),
            "sources": sources,
            "chunks": chunks,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "response": f"Error processing your query: {str(e)}",
            "sources": [],
            "chunks": [],
            "processing_time": 0
        }

def generate_accuracy_percentage():
    """Generate a simulated accuracy percentage for the response"""
    import random
    return random.randint(85, 99)
