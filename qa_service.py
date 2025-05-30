import time
import uuid
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import google.genai as genai
from pathlib import Path
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool, EmbedContentConfig

import streamlit as st
from config import VECTOR_STORE_PATH
from vector_db import initialize_vector_db, check_db_status
from gemini_embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)

class QAService:
    def __init__(self):
        """Initialize QA service with Gemini client and vector store"""
        try:
            # Initialize Gemini client with API key
            self.client = genai.Client(api_key=st.secrets["api_keys"]["gemini"])
            self.model_name = st.secrets["settings"]["default_model"]
            self.embedding_model = "embedding-001"
            self.embedding_dimension = 768  # Gemini embeddings have 768 dimensions by default
            
            # Initialize embeddings
            self.embeddings = GeminiEmbeddings(api_key=st.secrets["api_keys"]["gemini"])
            
            # Initialize vector store
            self.vector_store = initialize_vector_db()
            if not self.vector_store:
                logger.warning("Vector store not initialized")
            
            # Initialize conversation history
            self.conversations: Dict[str, List[Dict]] = {}
            
            # Initialize with default prompts
            self.QA_TEMPLATE = """
            You are an AI assistant specialized in medical and scientific research. 
            Answer the user's question based on the provided context from research papers.
            
            {conversation_context}
            
            Context from relevant documents:
            {context}
            
            User Question: {question}
            
            Instructions:
            1. Answer the question based ONLY on the provided context.
            2. Consider the conversation history when interpreting the question.
            3. If the context doesn't contain enough information, say so clearly.
            4. Cite specific documents in your answer.
            5. Be concise and accurate.
            6. For medical topics, include a disclaimer about consulting healthcare professionals.
            
            Answer:
            """
            
            self.SEARCH_TEMPLATE = """
            You are an AI assistant specialized in medical and scientific research.
            Answer the user's question based on the search results provided.
            
            {conversation_context}
            
            User Question: {question}
            
            Instructions:
            1. Answer the question based on the search results.
            2. Consider the conversation history when interpreting the question.
            3. Be concise and accurate.
            4. Include relevant sources in your answer.
            5. For medical topics, include a disclaimer about consulting healthcare professionals.
            
            Answer:
            """
            
            self.COMBINED_TEMPLATE = """
            You are an AI assistant specialized in medical and scientific research.
            Answer the user's question based on both the provided research documents and web search results.
            
            {conversation_context}
            
            Context from research documents:
            {document_context}
            
            Context from web search:
            {web_context}
            
            User Question: {question}
            
            Instructions:
            1. Answer the question by combining information from both sources
            2. Prioritize information from research documents for established medical knowledge
            3. Use web search results for recent developments or additional context
            4. Clearly indicate which information comes from which source
            5. Be concise and accurate
            6. For medical topics, include a disclaimer about consulting healthcare professionals
            7. Cite sources appropriately using [doc1], [doc2], etc. for documents and [web1], [web2], etc. for web sources
            
            Answer:
            """
            
            self.QUALITY_CHECK_TEMPLATE = """
            You are an AI assistant specialized in evaluating the quality and completeness of medical and scientific answers.
            Evaluate the following answer for a medical/scientific question.
            
            Question: {question}
            Answer: {answer}
            
            Instructions:
            1. Evaluate if the answer is complete and accurate
            2. Check if it includes necessary medical disclaimers
            3. Assess if it has proper citations
            4. Determine if it needs more recent or comprehensive information
            5. Check if the answer explicitly states it cannot answer the question
            
            Provide your evaluation in JSON format with the following fields:
            {{
                "is_complete": boolean,
                "needs_web_search": boolean,
                "confidence_score": float (0-1),
                "missing_elements": list of strings,
                "reasoning": string
            }}
            
            Important: Set needs_web_search to true if:
            - The answer is missing critical safety information
            - The answer explicitly states it cannot answer the question
            - The answer indicates the provided context is insufficient
            
            For general completeness or additional details, set needs_web_search to false and list the missing elements.
            
            Return ONLY the JSON object, no other text.
            """
            
            logger.info("QA service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA service: {str(e)}")
            raise
    
    def _get_or_create_conversation(self, conversation_id: Optional[str] = None) -> tuple[str, List[Dict]]:
        """Get existing conversation or create new one"""
        if conversation_id and conversation_id in self.conversations:
            return conversation_id, self.conversations[conversation_id]
        
        new_id = str(uuid.uuid4())
        self.conversations[new_id] = []
        return new_id, self.conversations[new_id]
    
    def _format_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Format conversation history into a string for context"""
        if not conversation_history:
            return "No previous conversation."
        
        context = "Previous conversation:\n"
        for exchange in conversation_history:
            context += f"User: {exchange.get('user', '')}\n"
            context += f"Assistant: {exchange.get('assistant', '')}\n\n"
        
        return context
    
    async def _check_answer_quality(self, question: str, answer: str) -> Dict:
        """Check the quality of the generated answer"""
        try:
            # Format the quality check prompt
            prompt = self.QUALITY_CHECK_TEMPLATE.format(
                question=question,
                answer=answer
            )
            
            # Get evaluation from Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Clean and parse the JSON response
            response_text = response.text.strip()
            # Remove any markdown code block markers
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                evaluation = json.loads(response_text)
                logger.info(f"Answer quality evaluation: {evaluation}")
                
                # Validate required fields
                required_fields = ["is_complete", "needs_web_search", "confidence_score", "missing_elements", "reasoning"]
                for field in required_fields:
                    if field not in evaluation:
                        evaluation[field] = self._get_default_quality_field(field)
                
                # Ensure correct types
                evaluation["is_complete"] = bool(evaluation["is_complete"])
                evaluation["needs_web_search"] = bool(evaluation["needs_web_search"])
                evaluation["confidence_score"] = float(evaluation["confidence_score"])
                evaluation["missing_elements"] = list(evaluation["missing_elements"])
                evaluation["reasoning"] = str(evaluation["reasoning"])
                
                return evaluation
                
            except json.JSONDecodeError as je:
                logger.error(f"JSON parsing error: {str(je)}, Response text: {response_text}")
                return self._get_default_quality_check()
            
        except Exception as e:
            logger.error(f"Error in answer quality check: {str(e)}")
            return self._get_default_quality_check()
    
    def _get_default_quality_field(self, field: str) -> Any:
        """Get default value for a quality check field"""
        defaults = {
            "is_complete": True,
            "needs_web_search": False,
            "confidence_score": 0.5,
            "missing_elements": [],
            "reasoning": "Default quality check"
        }
        return defaults.get(field, None)
    
    def _get_default_quality_check(self) -> Dict:
        """Get default quality check response"""
        return {
            "is_complete": True,
            "needs_web_search": False,
            "confidence_score": 0.5,
            "missing_elements": ["Quality check failed"],
            "reasoning": "Default quality check due to error"
        }
    
    async def _get_web_search_results(self, question: str, conversation_context: str) -> tuple[str, List[Dict]]:
        """Get results from web search"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = self.SEARCH_TEMPLATE.format(
                    question=question,
                    conversation_context=conversation_context
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerateContentConfig(
                        tools=[Tool(google_search=GoogleSearch())],
                        temperature=0.0
                    )
                )
                
                answer = response.text
                web_sources = []
                
                # Process sources from grounding metadata
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'grounding_metadata'):
                            web_sources.extend(self._process_grounding_metadata(candidate.grounding_metadata))
                
                return answer, web_sources
                
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    # Wait with exponential backoff before retrying
                    import time
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(f"Model overloaded (503), retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error in web search: {str(e)}")
                return "", []
    
    def _process_grounding_metadata(self, grounding_metadata) -> List[Dict]:
        """Process grounding metadata from web search results"""
        web_sources = []
        
        try:
            if not grounding_metadata:
                logger.warning("No grounding metadata provided")
                return web_sources
                
            # Extract sources from grounding chunks
            if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
                for chunk in grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web'):
                        # Get the original URL directly from the web object
                        original_url = chunk.web.uri
                        
                        # Skip if no URL
                        if not original_url:
                            continue
                            
                        # Extract domain from URL if not provided
                        domain = chunk.web.domain
                        if not domain:
                            from urllib.parse import urlparse
                            try:
                                parsed_url = urlparse(original_url)
                                domain = parsed_url.netloc
                                # Remove 'www.' prefix if present
                                if domain.startswith('www.'):
                                    domain = domain[4:]
                            except Exception as e:
                                logger.warning(f"Failed to parse URL: {str(e)}")
                                continue
                    
                        # Validate source title
                        title = chunk.web.title
                        if not title or len(title.strip()) == 0:
                            title = domain or "Unknown Source"
                    
                        # Add source with validation
                        if original_url and domain:
                            web_sources.append({
                                "source": original_url,
                                "title": title,
                                "relevance_score": 1.0,  # Default score for web sources
                                "chunk": chunk.web.snippet if hasattr(chunk.web, 'snippet') else "",
                                "metadata": {
                                    'source_type': 'web',
                                    'domain': domain,
                                    'is_valid': True
                                }
                            })
                        else:
                            logger.warning(f"Invalid source: {chunk.web.__dict__}")
            
            return web_sources
            
        except Exception as e:
            logger.error(f"Error processing grounding metadata: {str(e)}")
            return []
    
    async def _get_document_results(self, query_embedding: List[float], max_sources: int) -> tuple[str, List[Dict]]:
        """Get results from document search"""
        try:
            # Search documents using vector store
            if not self.vector_store:
                logger.warning("Vector store not initialized. Using empty context.")
                return "", []
            
            # Try to use similarity_search_with_relevance_scores if available
            if hasattr(self.vector_store, "similarity_search_with_relevance_scores"):
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query_embedding,
                    k=max_sources
                )
                # results: List[Tuple[Document, float]]
                doc_matches = [(doc.metadata.get("source", "Unknown"), doc.page_content, score)
                               for doc, score in results]
            else:
                # Fallback: use similarity_search and assign a default score
                docs = self.vector_store.similarity_search(
                    query_embedding,
                    k=max_sources
                )
                doc_matches = [(doc.metadata.get("source", "Unknown"), doc.page_content, 1.0)
                               for doc in docs]
            
            # Build context and sources
            context = ""
            sources = []
            
            for doc_id, chunk, score in doc_matches:
                sources.append({
                    "source": doc_id,
                    "relevance_score": float(score),
                    "chunk": chunk[:500] + '...' if len(chunk) > 500 else chunk,
                    "metadata": {
                        "source_type": "document",
                        "is_valid": True
                    }
                })
                context += f"\n\nDocument: {doc_id}\n{chunk}"
            
            return context, sources
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return "", []
    
    async def process_query(self, query: str, conversation_id: Optional[str] = None, max_sources: int = 5, prompt_template: Optional[str] = None) -> Dict:
        """Process a query and generate a response. Accepts an optional prompt_template for custom prompts."""
        start_time = time.time()
        sources = []
        
        # Get or create conversation memory
        conversation_id, conversation_history = self._get_or_create_conversation(conversation_id)
        conversation_context = self._format_conversation_context(conversation_history)
        
        try:
            # Get query embedding using the embedding model
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=query,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.embedding_dimension
                    )
                )
                query_embedding = response.embeddings[0].values
            except Exception as e:
                if "503" in str(e):
                    logger.error("Model overloaded (503), please try again later")
                    return {
                        "answer": "I apologize, but the AI model is currently overloaded. Please try again in a few moments.",
                        "sources": [],
                        "conversation_id": conversation_id,
                        "processing_time": time.time() - start_time,
                        "created_at": datetime.utcnow().isoformat(),
                        "metadata": {
                            "error": "Model overloaded",
                            "debug_info": {
                                "error_type": "ModelOverloaded",
                                "error_message": str(e)
                            }
                        }
                    }
                logger.error(f"Error getting query embedding: {str(e)}")
                return {
                    "answer": "I apologize, but I encountered an error while processing your query. Please try again later.",
                    "sources": [],
                    "conversation_id": conversation_id,
                    "processing_time": time.time() - start_time,
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": {
                        "error": "Embedding error",
                        "debug_info": {
                            "error_type": "EmbeddingError",
                            "error_message": str(e)
                        }
                    }
                }
            
            # Run document and web search in parallel
            doc_task = self._get_document_results(query_embedding, max_sources)
            web_task = self._get_web_search_results(query, conversation_context)
            
            doc_context, doc_sources = await doc_task
            web_answer, web_sources = await web_task
            
            # Combine sources
            sources.extend(doc_sources)
            sources.extend(web_sources)
            
            # Format the prompt using the provided template or the default combined template
            if prompt_template:
                prompt = prompt_template.format(
                    question=query,
                    conversation_context=conversation_context,
                    document_context=doc_context,
                    web_context=web_answer
                )
            else:
                prompt = self.COMBINED_TEMPLATE.format(
                    question=query,
                    conversation_context=conversation_context,
                    document_context=doc_context,
                    web_context=web_answer
                )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.95,
                    top_k=0,
                    max_output_tokens=2048
                )
            )
            answer = response.text
            
            try:
                # Check answer quality
                quality_check = await self._check_answer_quality(query, answer)
            except Exception as qe:
                logger.error(f"Error in quality check: {str(qe)}")
                quality_check = {
                    "is_complete": True,
                    "needs_web_search": False,
                    "confidence_score": 0.5,
                    "missing_elements": ["Quality check failed"],
                    "reasoning": f"Error in quality check: {str(qe)}"
                }
            
            # Update conversation history
            conversation_history.append({
                "user": query,
                "assistant": answer
            })
            
            return {
                "answer": answer,
                "sources": sources,
                "conversation_id": conversation_id,
                "processing_time": time.time() - start_time,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "quality_check": quality_check,
                    "source": "combined_search",
                    "debug_info": {
                        "document_sources": len(doc_sources),
                        "web_sources": len(web_sources),
                        "total_sources": len(sources)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try again later.",
                "sources": sources,
                "conversation_id": conversation_id,
                "processing_time": time.time() - start_time,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "error": str(e),
                    "debug_info": {
                        "sources_found": len(sources),
                        "error_type": type(e).__name__
                    }
                }
            }
    
    def get_conversation_history(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve conversation history"""
        if conversation_id not in self.conversations:
            return None
        
        history = self.conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "exchanges": history,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }

# Create a singleton instance
qa_service = QAService() 
