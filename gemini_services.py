import os
import logging
import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        """Initialize Gemini service with API key from secrets"""
        try:
            # Configure Gemini with API key
            genai.configure(api_key=st.secrets["api_keys"]["gemini"])
            
            # Initialize Gemini model
            self.model = genai.GenerativeModel(
                model_name=st.secrets["settings"]["default_model"],
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "top_k": 0,
                    "max_output_tokens": 2048
                }
            )
            
            # Store embedding model name
            self.embedding_model = st.secrets["settings"]["embedding_model"]
            
            # Initialize conversation history
            self.conversations: Dict[str, List[Dict]] = {}
            
            logger.info("Gemini service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini service: {str(e)}")
            raise
    
    # Templates for different types of prompts
    QA_TEMPLATE = """
    You are a medical research assistant. Use the following context to answer the question.
    If you cannot find the answer in the context, say so. Do not make up information.
    
    Previous conversation:
    {conversation_context}
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    QUALITY_CHECK_TEMPLATE = """
    Evaluate the following answer for completeness and accuracy:
    
    Question: {question}
    Answer: {answer}
    
    Provide a JSON response with the following structure:
    {{
        "is_complete": boolean,
        "confidence_score": float (0-1),
        "missing_information": string,
        "suggestions": string
    }}
    """
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        try:
            embeddings = []
            for text in texts:
                embedding = self.model.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding.embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            # Return random embeddings as fallback
            return [[0.0] * 768 for _ in texts]
    
    def create_query_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text (query)"""
        try:
            embedding = self.model.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_query"
            )
            return embedding.embedding
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            # Return random embedding as fallback
            return [0.0] * 768
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Gemini model"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."
    
    def check_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """Check the quality of an answer"""
        try:
            prompt = self.QUALITY_CHECK_TEMPLATE.format(
                question=question,
                answer=answer
            )
            response = self.generate_response(prompt)
            
            # Parse the JSON response
            quality_check = json.loads(response)
            return quality_check
        except Exception as e:
            logger.error(f"Error checking answer quality: {str(e)}")
            return {
                "is_complete": True,
                "confidence_score": 0.5,
                "missing_information": "Unable to check quality",
                "suggestions": "Please verify the answer manually"
            }

# Create a singleton instance
gemini_service = GeminiService() 