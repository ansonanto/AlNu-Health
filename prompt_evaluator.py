import os
import json
import time
import uuid
import logging
import streamlit as st
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Optional
from gemini_services import gemini_service
from qa_service import QAService, qa_service
from firebase_helpers import save_evaluation_firestore, fetch_evaluations_from_firestore

from config import GEMINI_API_KEY, MODEL_NAME, PROMPTS_DIR, EVALUATIONS_DIR, GEMINI_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROMPTS_DIRECTORY = PROMPTS_DIR
EVALUATIONS_DIRECTORY = EVALUATIONS_DIR

# Use the combined template as the default prompt
DEFAULT_SYSTEM_PROMPT = QAService().COMBINED_TEMPLATE

def ensure_directories():
    """Ensure necessary directories exist"""
    os.makedirs(PROMPTS_DIRECTORY, exist_ok=True)
    os.makedirs(EVALUATIONS_DIRECTORY, exist_ok=True)

def save_prompt(name, prompt_text):
    """Save a prompt template to file"""
    ensure_directories()
    prompt_id = str(uuid.uuid4())
    prompt_data = {
        "id": prompt_id,
        "name": name,
        "text": prompt_text,
        "created_at": datetime.now().isoformat()
    }
    
    filename = os.path.join(PROMPTS_DIRECTORY, f"{prompt_id}.json")
    with open(filename, 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    return prompt_id

def delete_prompt(prompt_id, force=False):
    """Delete a prompt template by its ID"""
    ensure_directories()
    prompt_path = os.path.join(PROMPTS_DIRECTORY, f"{prompt_id}.json")
    
    # First check if there are any evaluations using this prompt
    evaluations = load_evaluations(prompt_id)
    if evaluations and not force:
        logger.warning(f"Cannot delete prompt {prompt_id} because it has {len(evaluations)} evaluations")
        return False, f"Cannot delete: This prompt has {len(evaluations)} evaluations. Delete them first or use force delete."
    
    # If force is true, delete all evaluations first
    if evaluations and force:
        for eval_data in evaluations:
            delete_evaluation(eval_data['id'])
        logger.info(f"Deleted {len(evaluations)} evaluations for prompt {prompt_id}")
    
    # Proceed with prompt deletion
    if os.path.exists(prompt_path):
        try:
            os.remove(prompt_path)
            logger.info(f"Deleted prompt {prompt_id}")
            return True, "Prompt and all associated evaluations deleted successfully!"
        except Exception as e:
            logger.error(f"Error deleting prompt {prompt_id}: {str(e)}")
            return False, f"Error: {str(e)}"
    else:
        logger.warning(f"Prompt {prompt_id} not found")
        return False, "Prompt not found."

def load_prompts():
    """Load all saved prompts"""
    ensure_directories()
    prompts = []
    
    for filename in os.listdir(PROMPTS_DIRECTORY):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(PROMPTS_DIRECTORY, filename), 'r') as f:
                    prompt_data = json.load(f)
                    prompts.append(prompt_data)
            except Exception as e:
                logger.error(f"Error loading prompt {filename}: {str(e)}")
    
    # Sort by creation date (newest first)
    prompts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return prompts

def save_evaluation(prompt_id, prompt_name, query, context, response, score, feedback):
    """Save an evaluation of a prompt"""
    ensure_directories()
    eval_id = str(uuid.uuid4())
    eval_data = {
        "id": eval_id,
        "prompt_id": prompt_id,
        "prompt_name": prompt_name,
        "query": query,
        "context": context,
        "response": response,
        "score": score,
        "feedback": feedback,
        "created_at": datetime.now().isoformat()
    }
    
    filename = os.path.join(EVALUATIONS_DIRECTORY, f"{eval_id}.json")
    with open(filename, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    return eval_id

def delete_evaluation(eval_id):
    """Delete an evaluation by its ID"""
    ensure_directories()
    eval_path = os.path.join(EVALUATIONS_DIRECTORY, f"{eval_id}.json")
    
    if os.path.exists(eval_path):
        try:
            os.remove(eval_path)
            logger.info(f"Deleted evaluation {eval_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting evaluation {eval_id}: {str(e)}")
            return False
    else:
        logger.warning(f"Evaluation {eval_id} not found")
        return False

def load_evaluations(prompt_id=None):
    """Load evaluations, optionally filtered by prompt_id"""
    ensure_directories()
    evaluations = []
    
    for filename in os.listdir(EVALUATIONS_DIRECTORY):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(EVALUATIONS_DIRECTORY, filename), 'r') as f:
                    eval_data = json.load(f)
                    if prompt_id is None or eval_data.get('prompt_id') == prompt_id:
                        evaluations.append(eval_data)
            except Exception as e:
                logger.error(f"Error loading evaluation {filename}: {str(e)}")
    
    # Sort by creation date (newest first)
    evaluations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return evaluations

def test_prompt(prompt_text, query, context):
    """Test a prompt with a query and context"""
    try:
        # Create prompt template with default if none provided
        if not prompt_text:
            prompt_text = """
            You are an AI assistant specialized in medical and scientific research. 
            Answer the user's question based on the provided context from research papers.
            
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
            """
        
        prompt_template = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"]
        )
        
        # Initialize the Gemini LLM
        # Ensure we're explicitly using the Gemini API key
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment variables or secrets.")
            
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=GEMINI_CONFIG.get("temperature", 0.0),
            top_p=GEMINI_CONFIG.get("top_p", 0.95),
            # top_k must be positive, so we'll use 40 as a default value
            top_k=40,  # Default value that works well
            max_output_tokens=GEMINI_CONFIG.get("max_output_tokens", 2048),
            google_api_key=GEMINI_API_KEY
        )
        
        # Create chain using the new RunnableSequence pattern
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Execute chain
        start_time = time.time()
        response = chain.invoke({"context": context, "question": query})
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error testing prompt: {str(e)}")
        return {
            "response": f"Error: {str(e)}",
            "processing_time": 0
        }

def prompt_evaluator_ui():
    """UI for prompt testing and evaluation tab with tabs, prompt selection, test, and evaluation."""
    st.header("Prompt Testing & Evaluation")
    tabs = st.tabs(["Test Prompts", "View Evaluations"])

    # --- Test Prompts Tab ---
    with tabs[0]:
        st.subheader("Test Different Prompts")
        prompt_templates = load_prompts()
        template_names = ["Default"] + [p["name"] for p in prompt_templates]
        selected_template = st.selectbox("Select a prompt template", template_names)
        use_custom = st.checkbox("Use custom prompt")

        if use_custom:
            custom_template = st.text_area("Custom Prompt Template", value=DEFAULT_SYSTEM_PROMPT, height=200)
            template_text = custom_template
        else:
            if selected_template == "Default":
                template_text = DEFAULT_SYSTEM_PROMPT
            else:
                template_text = next((p["text"] for p in prompt_templates if p["name"] == selected_template), DEFAULT_SYSTEM_PROMPT)
            st.text_area("Prompt Template (Read Only)", template_text, height=200, disabled=True)

        st.subheader("Test Input")
        query = st.text_input("Query")

        if st.button("Test Prompt"):
            if query:
                with st.spinner("Testing prompt..."):
                    try:
                        # Use asyncio to run the async process_query
                        import asyncio
                        result = asyncio.run(qa_service.process_query(
                            query=query,
                            prompt_template=template_text
                        ))
                        st.session_state.prompt_test_result = result
                        st.session_state.show_evaluation = True
                        
                    except Exception as e:
                        st.error(f"Error testing prompt: {str(e)}")
                        logger.error(f"Error in prompt test: {str(e)}")
            else:
                st.warning("Please enter a query to test the prompt.")

        # Show test result and evaluation form
        if "prompt_test_result" in st.session_state and st.session_state.get("show_evaluation", False):
            result = st.session_state.prompt_test_result
            
            # Display the response
            st.markdown(f"**Response:**\n{result['answer']}")
            
            # Display sources if available
            if result.get('sources'):
                st.markdown("**Sources:**")
                for source in result['sources']:
                    st.markdown(f"- {source.get('title', source.get('source', 'Unknown'))}")
            
            st.caption(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            
            # Add rating and feedback section
            st.subheader("Rate the Response")
            rating = st.slider("How would you rate this response?", 1, 10, 5)
            feedback = st.text_area("Additional feedback (optional)", height=100)
            
            if st.button("Submit Evaluation"):
                # Save the evaluation to Firestore using logged-in user info
                user_id = st.session_state.get("user_id")
                evaluator_name = st.session_state.get("evaluator_name")
                id_token = st.session_state.get("id_token")
                prompt = template_text
                query_val = query
                response_val = result['answer']
                sources_val = result.get('sources', [])
                rating_val = rating
                feedback_val = feedback
                success = save_evaluation_firestore(
                    id_token,
                    user_id,
                    evaluator_name,
                    prompt,
                    query_val,
                    response_val,
                    sources_val,
                    rating_val,
                    feedback_val
                )
                if success:
                    st.success("Evaluation saved to Firestore!")
                else:
                    st.error("Failed to save evaluation to Firestore.")
                st.session_state.show_evaluation = False

        with st.expander("Instructions"):
            st.markdown("""
            ### How to Use
            1. Select a prompt template or use the default
            2. Optionally enable and edit a custom prompt
            3. Enter your query, then click "Test Prompt"
            4. Rate the response and provide feedback
            ### Tips
            - Be specific in your questions
            - Consider medical disclaimers
            - Cite your sources
            """)
            st.subheader("Example")
            st.markdown("""
            **Question:**
            What are the benefits of exercise for heart health?
            """)

    # --- View Evaluations Tab ---
    with tabs[1]:
        st.subheader("View Evaluations")
        id_token = st.session_state.get("id_token")
        user_id = st.session_state.get("user_id")
        if not id_token:
            st.info("Please log in to view evaluations.")
        else:
            # Fetch from Firestore (show all evaluations)
            evaluations = fetch_evaluations_from_firestore(id_token)
            if not evaluations:
                st.info("No evaluations found.")
            else:
                # Get unique evaluator names
                names = sorted(set(e.get("evaluator_name", "Unknown") for e in evaluations))
                selected_name = st.selectbox("Filter by evaluator name", ["All"] + names)
                if selected_name != "All":
                    filtered_evals = [e for e in evaluations if e.get("evaluator_name", "Unknown") == selected_name]
                else:
                    filtered_evals = evaluations
                for eval_data in filtered_evals:
                    with st.expander(f"{eval_data.get('evaluator_name', 'Unknown')} | {eval_data.get('query', '')[:40]}..."):
                        st.markdown(f"**Evaluator:** {eval_data.get('evaluator_name', '')}")
                        st.markdown(f"**Prompt:** {eval_data.get('prompt', '')}")
                        st.markdown(f"**Query:** {eval_data.get('query', '')}")
                        st.markdown(f"**Response:** {eval_data.get('response', '')}")
                        st.markdown(f"**Sources:** {eval_data.get('sources', '')}")
                        st.markdown(f"**Rating:** {eval_data.get('rating', '')}/10")
                        st.markdown(f"**Feedback:** {eval_data.get('feedback', '')}")
                        st.markdown(f"**Timestamp:** {eval_data.get('timestamp', '')}")

if __name__ == "__main__":
    prompt_evaluator_ui()
