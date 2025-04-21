"""
Query processor for LlamaIndex-based vector database implementation.
"""
import time
import logging
import streamlit as st
from typing import List, Dict, Any, Optional

from openai import OpenAI
from llama_index.core.response_synthesizers import get_response_synthesizer
# Use the OpenAI integration directly instead of the llama_index wrapper
from llama_index.llms.openai import OpenAI as LlamaOpenAI

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
        from llama_index.indices.query.schema import QueryMode
        
        # For LlamaIndex 0.9.48, we need to use different parameters
        query_engine = db.as_query_engine(
            similarity_top_k=top_k,
            # Use simpler response mode for compatibility
            response_mode="default" 
        )
        
        # Execute query
        logger.info(f"Executing query: {query}")
        response = query_engine.query(query)
        
        # Extract source nodes and their metadata
        # In 0.9.48, source_nodes is a list, not a property
        source_nodes = response.source_nodes if hasattr(response, "source_nodes") else []
        sources = []
        chunks = []
        
        for node in source_nodes:
            # Handle different node structure in 0.9.48
            if hasattr(node, "node"):
                # Newer versions wrap the node
                node_obj = node.node
                score = node.score if hasattr(node, "score") else None
            else:
                # Older versions have the node directly
                node_obj = node
                score = getattr(node, "score", None)
                
            # Extract metadata
            metadata = getattr(node_obj, "metadata", {})
            text = getattr(node_obj, "text", "")
                
            source_info = {
                "source": metadata.get("name", "Unknown"),
                "score": score,
                "chunk": metadata.get("chunk", 0),
                "total_chunks": metadata.get("total_chunks", 1)
            }
            sources.append(source_info)
            
            chunk_info = {
                "content": text,
                "metadata": metadata
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
