"""
Simple query processor for FAISS-based vector database implementation.
"""
import time
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
import random

from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_documents(query, db, conversation_history=None, is_summary_request=False):
    """Process query and generate response using RAG approach"""
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
        k = 3 if is_summary_request else 5
        
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Retrieve relevant documents
        try:
            # Try with relevance scores
            retrieval_results = db.similarity_search_with_relevance_scores(query, k=k)
            
            # Log the relevance scores for debugging
            logger.info(f"Query: {query}")
            logger.info(f"Retrieved {len(retrieval_results)} documents with scores:")
            for i, (doc, score) in enumerate(retrieval_results):
                logger.info(f"  Doc {i+1}: score={score:.4f}, source={doc['metadata'].get('source', 'Unknown')}")
                
        except Exception as e:
            # If that fails, fall back to regular similarity search
            logger.error(f"Error with similarity_search_with_relevance_scores: {str(e)}")
            logger.info("Falling back to regular similarity_search")
            
            docs = db.similarity_search(query, k=k)
            retrieval_results = [(doc, doc["metadata"].get("score", 0.0)) for doc in docs]
        
        # Extract sources and chunks
        sources = []
        chunks = []
        context_text = ""
        
        for i, (doc, score) in enumerate(retrieval_results):
            # Add to sources
            source_info = {
                "source": doc["metadata"].get("source", "Unknown"),
                "score": score,
                "chunk": i,
                "total_chunks": len(retrieval_results)
            }
            sources.append(source_info)
            
            # Add to chunks
            chunk_info = {
                "content": doc["content"],
                "metadata": doc["metadata"]
            }
            chunks.append(chunk_info)
            
            # Add to context text
            context_text += f"\nSource {i+1} ({doc['metadata'].get('source', 'Unknown')}):\n{doc['content']}\n"
        
        # Generate response using OpenAI
        if not context_text:
            response_text = "I couldn't find any relevant information to answer your question."
        else:
            # Create prompt
            if is_summary_request:
                prompt = f"""You are a medical research assistant. Based on the following research paper excerpts, provide a concise summary:
                
{context_text}

Summary:"""
            else:
                prompt = f"""You are a medical research assistant. Answer the question based only on the following research paper excerpts. 
If the information to answer the question is not contained in the excerpts, say "I don't have enough information to answer this question."

{context_text}

Question: {query}
Answer:"""
            
            # Generate response
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful medical research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                response_text = f"Error generating response: {str(e)}"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        # Return the response and metadata
        return {
            "response": response_text,
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
    return random.randint(85, 99)
