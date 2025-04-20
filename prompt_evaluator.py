import os
import json
import time
import uuid
import logging
import streamlit as st
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any, Optional

from config import OPENAI_API_KEY, MODEL_NAME, PROMPTS_DIR, EVALUATIONS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROMPTS_DIRECTORY = PROMPTS_DIR
EVALUATIONS_DIRECTORY = EVALUATIONS_DIR
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specialized in medical and scientific research. 
Answer the user's question based on the provided context from research papers.

Context from relevant documents:
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Cite the specific documents you're using in your answer.
4. Be concise and accurate.
5. If the question is about medical advice, remind the user that you're providing information from research papers, not personalized medical advice.

Answer:
"""

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

def delete_prompt(prompt_id):
    """Delete a prompt template by its ID"""
    ensure_directories()
    prompt_path = os.path.join(PROMPTS_DIRECTORY, f"{prompt_id}.json")
    
    # First check if there are any evaluations using this prompt
    evaluations = load_evaluations(prompt_id)
    if evaluations:
        logger.warning(f"Cannot delete prompt {prompt_id} because it has {len(evaluations)} evaluations")
        return False, f"Cannot delete: This prompt has {len(evaluations)} evaluations. Delete them first."
    
    # If no evaluations, proceed with deletion
    if os.path.exists(prompt_path):
        try:
            os.remove(prompt_path)
            logger.info(f"Deleted prompt {prompt_id}")
            return True, "Prompt deleted successfully!"
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
        # Create prompt template
        prompt_template = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"]
        )
        
        # Initialize the LLM
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Execute chain
        start_time = time.time()
        response = chain.run(context=context, question=query)
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
    """Streamlit interface for prompt testing and evaluation"""
    st.title("Prompt Testing & Evaluation")
    
    tabs = st.tabs(["Test Prompts", "Manage Prompts", "View Evaluations"])
    
    with tabs[0]:  # Test Prompts
        st.header("Test Different Prompts")
        
        # Load saved prompts
        prompts = load_prompts()
        prompt_options = ["Default"] + [p["name"] for p in prompts]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_prompt = st.selectbox("Select a prompt template", prompt_options)
        
        with col2:
            use_custom = st.checkbox("Use custom prompt")
        
        # Get the prompt text
        if use_custom:
            prompt_text = st.text_area("Custom Prompt Template", DEFAULT_SYSTEM_PROMPT, height=300)
            prompt_name = "Custom"
            prompt_id = None
        else:
            if selected_prompt == "Default":
                prompt_text = DEFAULT_SYSTEM_PROMPT
                prompt_name = "Default"
                prompt_id = None
            else:
                selected_idx = prompt_options.index(selected_prompt) - 1  # Adjust for "Default"
                prompt_text = prompts[selected_idx]["text"]
                prompt_name = prompts[selected_idx]["name"]
                prompt_id = prompts[selected_idx]["id"]
            
            st.text_area("Prompt Template (Read Only)", prompt_text, height=200, disabled=True)
        
        # Query and context inputs
        st.subheader("Test Input")
        query = st.text_input("Query", "What are the latest treatments for type 2 diabetes?")
        context = st.text_area("Context (simulated retrieval results)", 
                              "Document: diabetes_treatment.pdf, Chunk: 3\n"
                              "Recent studies have shown that GLP-1 receptor agonists like semaglutide and tirzepatide "
                              "are highly effective for treating type 2 diabetes. They work by stimulating insulin "
                              "secretion, suppressing glucagon, and slowing gastric emptying. These medications have "
                              "shown significant benefits for weight loss and cardiovascular outcomes.\n\n"
                              "Document: diabetes_review.pdf, Chunk: 7\n"
                              "SGLT-2 inhibitors represent another important class of medications for type 2 diabetes. "
                              "They work by preventing glucose reabsorption in the kidneys, leading to increased "
                              "glucose excretion in urine. Clinical trials have demonstrated cardiovascular and renal "
                              "protective effects independent of their glucose-lowering action.",
                              height=200)
        
        # Use session state to store test results
        if 'test_result' not in st.session_state:
            st.session_state.test_result = None
        
        # Test button
        if st.button("Test Prompt"):
            with st.spinner("Testing prompt..."):
                result = test_prompt(prompt_text, query, context)
                st.session_state.test_result = result
                st.session_state.current_prompt_text = prompt_text
                st.session_state.current_prompt_name = prompt_name
                st.session_state.current_prompt_id = prompt_id
                st.session_state.current_query = query
                st.session_state.current_context = context
        
        # Display results if available
        if st.session_state.test_result:
            st.subheader("Response")
            st.write(st.session_state.test_result["response"])
            st.info(f"Processing time: {st.session_state.test_result['processing_time']:.2f} seconds")
            
            # Evaluation section
            st.subheader("Evaluate Response")
            score = st.slider("Score (1-10)", 1, 10, 7, key="eval_score")
            feedback = st.text_area("Feedback (optional)", "", key="eval_feedback")
            
            # For custom prompts
            if use_custom or st.session_state.current_prompt_id is None:
                custom_name = st.text_input("Name for this prompt template", "My Custom Prompt")
            
            # Save evaluation button (outside of any conditional)
            if st.button("Save Evaluation"):
                # If using custom prompt, save it first
                if use_custom or st.session_state.current_prompt_id is None:
                    prompt_id = save_prompt(custom_name, st.session_state.current_prompt_text)
                    prompt_name = custom_name
                else:
                    prompt_id = st.session_state.current_prompt_id
                    prompt_name = st.session_state.current_prompt_name
                
                # Save evaluation
                eval_id = save_evaluation(
                    prompt_id, 
                    prompt_name, 
                    st.session_state.current_query, 
                    st.session_state.current_context, 
                    st.session_state.test_result["response"], 
                    score, 
                    feedback
                )
                
                st.success(f"Evaluation saved with ID: {eval_id}")
    
    with tabs[1]:  # Manage Prompts
        st.header("Manage Prompt Templates")
        
        # Create new prompt
        st.subheader("Create New Prompt Template")
        new_prompt_name = st.text_input("Prompt Name", "My New Prompt")
        new_prompt_text = st.text_area("Prompt Template", DEFAULT_SYSTEM_PROMPT, height=300)
        
        if st.button("Save New Prompt"):
            prompt_id = save_prompt(new_prompt_name, new_prompt_text)
            st.success(f"Prompt saved with ID: {prompt_id}")
        
        # List existing prompts
        st.subheader("Existing Prompt Templates")
        prompts = load_prompts()
        
        if not prompts:
            st.info("No saved prompts yet. Create one above!")
        else:
            for prompt in prompts:
                with st.expander(f"{prompt['name']} ({prompt['created_at'][:10]})"):
                    # Create columns for layout - main content and delete button
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.text_area(f"Prompt Template", prompt["text"], height=200, key=prompt["id"], disabled=True)
                        
                        # Get evaluation stats for this prompt
                        evals = load_evaluations(prompt["id"])
                        if evals:
                            avg_score = sum(e["score"] for e in evals) / len(evals)
                            st.info(f"Average score: {avg_score:.1f}/10 from {len(evals)} evaluations")
                    
                    with col2:
                        # Add delete button
                        prompt_id = prompt["id"]
                        delete_key = f"delete_prompt_{prompt_id}"
                        
                        if st.button("🗑️ Delete", key=delete_key):
                            success, message = delete_prompt(prompt_id)
                            if success:
                                st.success(message)
                                st.rerun()  # Refresh the page to update the list
                            else:
                                st.error(message)
    
    with tabs[2]:  # View Evaluations
        st.header("View Evaluations")
        
        # Filter options
        prompt_names = [p["name"] for p in prompts]
        filter_prompt = st.selectbox(
            "Filter by prompt", 
            ["All Prompts"] + prompt_names
        )
        
        # Load evaluations with filter
        if filter_prompt == "All Prompts":
            evaluations = load_evaluations()
        else:
            selected_idx = prompt_names.index(filter_prompt)
            prompt_id = prompts[selected_idx]["id"]
            evaluations = load_evaluations(prompt_id)
        
        if not evaluations:
            st.info("No evaluations found with the current filter.")
        else:
            st.info(f"Found {len(evaluations)} evaluations")
            
            # Display evaluations
            for eval_data in evaluations:
                with st.expander(f"{eval_data['prompt_name']} - Score: {eval_data['score']}/10 ({eval_data['created_at'][:10]})"):
                    # Create columns for layout - main content and delete button
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.subheader("Query")
                        st.write(eval_data["query"])
                        
                        st.subheader("Context")
                        st.text(eval_data["context"])
                        
                        st.subheader("Response")
                        st.write(eval_data["response"])
                        
                        if eval_data["feedback"]:
                            st.subheader("Feedback")
                            st.write(eval_data["feedback"])
                    
                    with col2:
                        # Add delete button
                        eval_id = eval_data["id"]
                        delete_key = f"delete_{eval_id}"
                        
                        if st.button("🗑️ Delete", key=delete_key):
                            if delete_evaluation(eval_id):
                                st.success("Evaluation deleted successfully!")
                                st.rerun()  # Refresh the page to update the list
                            else:
                                st.error("Failed to delete evaluation.")

if __name__ == "__main__":
    prompt_evaluator_ui()
