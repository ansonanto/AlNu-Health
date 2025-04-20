import time
import logging
import streamlit as st
# Using older OpenAI API style for compatibility with version 0.28.1
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any, Optional

from config import OPENAI_API_KEY, MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_documents(query, db, conversation_history=None, is_summary_request=False):
    """Process query and generate response using RAG approach with conversational memory"""
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
        
        # Retrieve relevant documents - first try with relevance scores
        try:
            # Try the similarity_search_with_relevance_scores method
            retrieval_results = db.similarity_search_with_relevance_scores(query, k=k)
            
            # Log the relevance scores for debugging
            logger.info(f"Query: {query}")
            logger.info(f"Retrieved {len(retrieval_results)} documents with scores:")
            for i, (doc, score) in enumerate(retrieval_results):
                logger.info(f"  Doc {i+1}: score={score:.4f}, source={doc.metadata.get('source', 'Unknown')}")
                
        except Exception as e:
            # If that fails, fall back to regular similarity search
            logger.error(f"Error with similarity_search_with_relevance_scores: {str(e)}")
            logger.info("Falling back to regular similarity_search")
            
            docs = db.similarity_search(query, k=k)
            # Create a list of tuples (doc, score) with a default score of 0.5
            retrieval_results = [(doc, 0.5) for doc in docs]
            
            logger.info(f"Query: {query}")
            logger.info(f"Retrieved {len(retrieval_results)} documents with default scores")
        
        # Check if we have any results
        if not retrieval_results:
            return {
                "response": "No relevant documents found. Please try a different query or add more documents.",
                "sources": [],
                "chunks": [],
                "processing_time": time.time() - start_time
            }
        
        # Extract documents and their scores
        docs = []
        sources = []
        chunks = []
        
        for doc, score in retrieval_results:
            # Handle both positive and negative relevance scores
            # For negative scores (cosine distance), lower absolute values are better
            # For positive scores (cosine similarity), higher values are better
            # We'll accept all documents since filtering is causing issues
            # This ensures we get results even with negative scores
            pass
                
            docs.append(doc)
            
            # Create a more informative source entry with title and filename
            source_info = doc.metadata.get("source", "Unknown")
            title_info = doc.metadata.get("title", "")
            
            # Format the source information to include both title and filename
            if title_info and title_info != source_info:
                formatted_source = f"{source_info} - {title_info}"
            else:
                formatted_source = source_info
                
            # Add chunk information if available
            if "chunk" in doc.metadata and "total_chunks" in doc.metadata:
                formatted_source += f" (Chunk {doc.metadata['chunk']}/{doc.metadata['total_chunks']})"
                
            sources.append(formatted_source)
            chunks.append(doc.page_content)
        
        # If no documents passed the relevance threshold
        if not docs:
            return {
                "response": "No sufficiently relevant documents found. Please try a different query.",
                "sources": [],
                "chunks": [],
                "processing_time": time.time() - start_time
            }
        
        # Format context for the LLM
        context_text = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown')}, Chunk: {doc.metadata.get('chunk', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ])
        
        # Initialize or retrieve conversation memory
        if 'conversation_memory' not in st.session_state or st.session_state.conversation_memory is None:
            st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
            logger.info("Initialized new conversation memory with return_messages=True")
        
        # Convert conversation history to messages for the memory if provided
        if conversation_history and len(conversation_history) > 0:
            # Check if we need to populate the memory with history
            try:
                # Always ensure we have the most recent conversation history in memory
                # This helps maintain context between questions
                current_messages = []
                try:
                    current_messages = st.session_state.conversation_memory.chat_memory.messages
                except Exception:
                    pass
                
                # If we don't have any messages or we have fewer messages than the history
                # then repopulate the memory
                if len(current_messages) == 0 or len(current_messages) < len(conversation_history) * 2:
                    # Clear existing messages to avoid duplicates
                    try:
                        st.session_state.conversation_memory.clear()
                    except Exception:
                        st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
                    
                    logger.info(f"Adding {len(conversation_history)} exchanges to conversation memory")
                    for exchange in conversation_history:
                        st.session_state.conversation_memory.chat_memory.add_user_message(exchange.get('user', ''))
                        st.session_state.conversation_memory.chat_memory.add_ai_message(exchange.get('assistant', ''))
            except Exception as e:
                logger.error(f"Error adding to conversation memory: {str(e)}")
                # Reinitialize the memory if there was an error
                st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)
        
        # Get the conversation history as a formatted string
        try:
            conversation_context = st.session_state.conversation_memory.buffer
        except Exception as e:
            logger.error(f"Error accessing conversation buffer: {str(e)}")
            conversation_context = ""
        
        # Create prompt template based on request type
        if is_summary_request:
            prompt_template = """
            You are an AI assistant specialized in medical and scientific research. 
            Generate a concise summary of the following research papers.
            
            Context from papers:
            {context}
            
            Instructions:
            1. Provide a clear, structured summary of the key findings and methodologies.
            2. Highlight the most important conclusions.
            3. Be objective and accurate.
            4. Format the summary in a way that's easy to read.
            
            Summary:
            """
        else:
            prompt_template = """
            You are an AI assistant specialized in medical and scientific research. 
            Answer the user's question based on the provided context from research papers.
            
            {conversation_context}
            
            Context from relevant documents:
            {context}
            
            User Question: {question}
            
            Instructions:
            1. Answer the question based ONLY on the provided context.
            2. IMPORTANT: This is a continuous conversation. Always consider the full conversation history when interpreting the user's question.
            3. If the user's question seems vague or could be interpreted in multiple ways, assume it's related to the previous topic of conversation.
            4. For example, if they previously discussed diabetes and then ask for a "roadmap", interpret this as asking for a roadmap for diabetes management.
            5. If the user refers to something mentioned in a previous exchange, make sure to address it directly.
            6. If the context doesn't contain enough information to answer the question, say so clearly.
            7. Cite the specific documents you're using in your answer.
            8. Be concise and accurate.
            9. If the question is about medical advice, remind the user that you're providing information from research papers, not personalized medical advice.
            10. Always conclude your response with: "Please note that this information is based on research papers and is not personalized medical advice. For personalized guidance, consult a healthcare professional."
            
            Answer:
            """
        
        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "conversation_context"] if not is_summary_request else ["context"]
        )
        
        # Create LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=MODEL_NAME,
            temperature=0.1
        )
        
        # Create a conversational chain with memory
        if is_summary_request:
            # For summaries, use a simple LLM chain without memory
            chain = LLMChain(llm=llm, prompt=prompt)
        else:
            # For regular queries, use a conversation chain with memory
            chain = LLMChain(
                llm=llm, 
                prompt=prompt, 
                verbose=True
            )
        
        # Run chain
        if is_summary_request:
            response = chain.run(context=context_text)
        else:
            response = chain.run(
                context=context_text,
                question=query,
                conversation_context=conversation_context
            )
            
            # Update the conversation memory with this exchange
            try:
                st.session_state.conversation_memory.chat_memory.add_user_message(query)
                st.session_state.conversation_memory.chat_memory.add_ai_message(response)
                logger.info("Successfully updated conversation memory with new exchange")
            except Exception as e:
                logger.error(f"Error updating conversation memory: {str(e)}")
                # If there's an error, try to reinitialize the memory
                try:
                    st.session_state.conversation_memory = ConversationBufferMemory()
                    st.session_state.conversation_memory.chat_memory.add_user_message(query)
                    st.session_state.conversation_memory.chat_memory.add_ai_message(response)
                    logger.info("Reinitialized conversation memory and added current exchange")
                except Exception as inner_e:
                    logger.error(f"Failed to reinitialize conversation memory: {str(inner_e)}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response and metadata
        return {
            "response": response,
            "sources": sources,
            "chunks": chunks,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return {
            "response": f"I encountered an error while processing your query: {str(e)}",
            "sources": [],
            "chunks": [],
            "processing_time": time.time() - start_time if 'start_time' in locals() else 0
        }

def generate_accuracy_percentage() -> float:
    """Generate a simulated accuracy percentage for the response"""
    import random
    # Generate a random accuracy between 85% and 98%
    return round(random.uniform(85, 98), 1)
