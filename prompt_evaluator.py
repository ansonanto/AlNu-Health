import os
import json
import time
import uuid
import logging
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException
from datetime import datetime
from gemini_services import gemini_service
from qa_service import QAService, qa_service
from firebase_helpers import save_prompt_firestore, load_prompts_firestore, save_evaluation_firestore, fetch_evaluations_from_firestore, delete_evaluation_firestore
import hashlib

from config import GEMINI_API_KEY, MODEL_NAME, PROMPTS_DIR, EVALUATIONS_DIR, GEMINI_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PROMPTS_DIRECTORY = PROMPTS_DIR
EVALUATIONS_DIRECTORY = EVALUATIONS_DIR

# Use the combined template as the default prompt
DEFAULT_SYSTEM_PROMPT = QAService().COMBINED_TEMPLATE

PROJECT_ID = st.secrets["firebase"]["projectId"]

def prompt_evaluator_ui():
    """UI for prompt testing and evaluation tab with tabs, prompt selection, test, and evaluation."""
    st.header("Prompt Testing & Evaluation")
    
    # Configure retry strategy and create session at the start
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    
    tabs = st.tabs(["Test Prompts", "View Evaluations", "Persona Batch Evaluator", "View Batch Results"])

    # --- Test Prompts Tab ---
    with tabs[0]:
        st.subheader("Test Different Prompts")
        
        st.markdown("---")
        st.subheader("Prompt Testing")
        id_token = st.session_state.get("id_token")
        user_id = st.session_state.get("user_id")
        user_name = st.session_state.get("evaluator_name")
        use_custom = st.checkbox("Use custom prompt")
        
        if use_custom:
            new_prompt_name = st.text_input("New Prompt Name", value="", key="new_prompt_name")
            custom_template = st.text_area("Custom Prompt Template", value=DEFAULT_SYSTEM_PROMPT, height=200, key="custom_prompt_text")
            template_text = custom_template
            prompt_id = None  # Not needed for custom prompt until saved
            if st.button("Save Custom Prompt"):
                if new_prompt_name.strip() == "":
                    st.warning("Please enter a name for your custom prompt.")
                elif not id_token or not user_id or not user_name:
                    st.warning("You must be logged in to save prompts.")
                else:
                    ok, prompt_id = save_prompt_firestore(id_token, user_id, user_name, new_prompt_name, custom_template)
                    if ok:
                        st.success(f"Custom prompt '{new_prompt_name}' saved!")
                        st.rerun()
                    else:
                        st.error("Failed to save prompt to Firestore.")
        else:
            prompt_templates = load_prompts_firestore(id_token) if id_token else []
            template_names = ["Default"] + [p["name"] for p in prompt_templates]
            selected_template = st.selectbox("Select a prompt template", template_names)
            if selected_template == "Default":
                template_text = DEFAULT_SYSTEM_PROMPT
                prompt_id = None
            else:
                selected_prompt = next((p for p in prompt_templates if p["name"] == selected_template), None)
                template_text = selected_prompt["text"] if selected_prompt else DEFAULT_SYSTEM_PROMPT
                prompt_id = selected_prompt["id"] if selected_prompt else None
            st.text_area("Prompt Template (Read Only)", template_text, height=200, disabled=True)
        
        st.subheader("Test Input")
        query = st.text_input("Query")

        if st.button("Test Prompt"):
            if query:
                with st.spinner("Testing prompt..."):
                    try:
                        import asyncio
                        result = asyncio.run(qa_service.process_query(
                            query=query,
                            prompt_template=template_text
                        ))
                        st.session_state.prompt_test_result = result
                        st.session_state.show_evaluation = True
                        # For custom prompt, use the new prompt name as the name, and prompt_id as None
                        if use_custom:
                            st.session_state.prompt_id_for_eval = None
                            st.session_state.prompt_name_for_eval = new_prompt_name
                        else:
                            st.session_state.prompt_id_for_eval = prompt_id
                            st.session_state.prompt_name_for_eval = selected_template if not use_custom else new_prompt_name
                    except Exception as e:
                        st.error(f"Error testing prompt: {str(e)}")
                        logger.error(f"Error in prompt test: {str(e)}")
            else:
                st.warning("Please enter a query to test the prompt.")

        # Show test result and evaluation form
        if "prompt_test_result" in st.session_state and st.session_state.get("show_evaluation", False):
            result = st.session_state.prompt_test_result
            prompt_id_for_eval = st.session_state.get("prompt_id_for_eval")
            prompt_name_for_eval = st.session_state.get("prompt_name_for_eval")

            st.markdown("---")
            st.markdown("### Response")
            st.write(result['answer'])

            # Display sources as a bulleted list
            if result.get('sources'):
                st.markdown("### Sources")
                sources = result['sources']
                if isinstance(sources, list):
                    for i, source in enumerate(sources, 1):
                        title = source.get('title') or source.get('source', 'Unknown')
                        chunk = source.get('chunk', '')
                        with st.expander(f"{title}"):
                            st.markdown(f"**Snippet:**\n\n{chunk}")
                else:
                    st.markdown(f"- {sources}")

            st.caption(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            st.markdown("---")

            # Rate the response
            st.markdown("### Rate the Response")
            col1, col2 = st.columns([2, 3])
            with col1:
                rating = st.slider("How would you rate this response?", 1, 10, 5)
            with col2:
                feedback = st.text_area("Additional feedback (optional)", height=100)

            if st.button("Submit Evaluation"):
                user_id = st.session_state.get("user_id")
                evaluator_name = st.session_state.get("evaluator_name")
                id_token = st.session_state.get("id_token")
                prompt = template_text
                query_val = query
                response_val = result['answer']
                sources_val = result.get('sources', [])
                rating_val = rating
                feedback_val = feedback
                ok = save_evaluation_firestore(
                    id_token,
                    user_id,
                    evaluator_name,
                    prompt_id_for_eval,
                    prompt,
                    query_val,
                    response_val,
                    sources_val,
                    rating_val,
                    feedback_val
                )
                if ok:
                    st.success("Evaluation saved to Firestore!")
                else:
                    st.error("Failed to save evaluation to Firestore.")
                st.session_state.show_evaluation = False
            st.markdown("---")

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
                expander_label = f"{eval_data.get('evaluator_name', 'Unknown')} | {eval_data.get('query', '')[:40]}..."
                with st.expander(expander_label):
                    evaluator = eval_data.get('evaluator_name', '')
                    prompt_text = eval_data.get('prompt', '')
                    st.markdown(f"**Evaluator:** {evaluator}")
                    st.code(prompt_text, language="")
                    st.markdown("#### Query")
                    st.info(eval_data.get('query', ''))
                    st.markdown("#### Response")
                    st.write(eval_data.get('response', ''))
                    st.markdown("#### Sources")
                    import ast
                    sources = eval_data.get('sources', '')
                    parsed_sources = []
                    if isinstance(sources, list):
                        parsed_sources = sources
                    elif isinstance(sources, str):
                        try:
                            parsed_sources = ast.literal_eval(sources)
                        except Exception:
                            parsed_sources = []
                    if isinstance(parsed_sources, list) and parsed_sources and isinstance(parsed_sources[0], dict):
                        for s in parsed_sources:
                            title = s.get('title') or s.get('source', 'Unknown')
                            chunk = s.get('chunk', '')
                            source_type = s.get('metadata', {}).get('source_type', '')
                            source_url = s.get('source', '')
                            if source_type == 'web' and source_url.startswith('http'):
                                display_title = title if title != source_url else s.get('metadata', {}).get('domain', title)
                                st.markdown(f"**[{display_title}]({source_url})**")
                            else:
                                st.markdown(f"**{title}**")
                            if chunk:
                                st.markdown(f"> {chunk}")
                    elif isinstance(parsed_sources, list) and parsed_sources:
                        for s in parsed_sources:
                            st.markdown(f"- {s}")
                    else:
                        st.info("No structured sources available for this evaluation.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rating", f"{eval_data.get('rating', '')}/10")
                    with col2:
                        st.caption(f"Timestamp: {eval_data.get('timestamp', '')}")
                    if eval_data.get('feedback', ''):
                        st.markdown("#### Feedback")
                        st.success(eval_data.get('feedback', ''))
                    st.markdown("---")
                    if st.button("🗑️ Delete Evaluation", key=f"delete_eval_{expander_label}"):
                        if 'document_id' in eval_data:
                            ok = delete_evaluation_firestore(id_token, eval_data['document_id'])
                            if ok:
                                st.success("Evaluation deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete evaluation.")
                        else:
                            st.error("Cannot delete: Firestore document ID not available.")

    # --- Persona Batch Evaluator Tab ---
    with tabs[2]:
        import pandas as pd
        import csv
        from collections import defaultdict
        # Load RLHF_Questions.csv
        questions = []
        personas = []
        persona_map = {}
        with open("RLHF/RLHF_Questions.csv", newline='') as csvfile:
            reader = list(csv.reader(csvfile))
            # Find header rows
            question_section = False
            persona_section = False
            for row in reader:
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                if 'Category' in row and 'Question' in row:
                    question_section = True
                    persona_section = False
                    continue
                if 'Persona' in row and 'Description' in row:
                    persona_section = True
                    question_section = False
                    continue
                if question_section and len(row) >= 2 and row[0].strip() != '':
                    # Some rows may have only 2 columns if question has a comma
                    cat = row[0].strip()
                    q = row[1].strip() if len(row) > 1 else ''
                    questions.append({'category': cat, 'question': q})
                if persona_section and len(row) >= 2 and row[0].strip() != '':
                    persona_id = row[0].strip()
                    desc = row[1].strip()
                    personas.append(desc)
                    persona_map[desc] = persona_id
        # st.write("Number of questions before deduplication:", len(questions))
        # st.write("Questions before deduplication:", questions)
        # # Deduplicate questions
        # questions = [dict(t) for t in {tuple(d.items()) for d in questions}]
        # st.write("Number of questions after deduplication:", len(questions))
        # st.write("Questions after deduplication:", questions)
        # st.write("Personas parsed:", personas)
        # Prompt selection/creation
        st.subheader("Select or Create Prompt")
        id_token = st.session_state.get("id_token")
        user_id = st.session_state.get("user_id")
        user_name = st.session_state.get("evaluator_name")
        prompt_templates = load_prompts_firestore(id_token) if id_token else []
        template_names = ["Default"] + [p["name"] for p in prompt_templates]
        selected_template = st.selectbox("Select a prompt template", template_names, key="batch_prompt_select")
        if selected_template == "Default":
            template_text = DEFAULT_SYSTEM_PROMPT
            prompt_id = None
        else:
            selected_prompt = next((p for p in prompt_templates if p["name"] == selected_template), None)
            template_text = selected_prompt["text"] if selected_prompt else DEFAULT_SYSTEM_PROMPT
            prompt_id = selected_prompt["id"] if selected_prompt else None
        use_custom = st.checkbox("Use custom prompt for batch", key="batch_custom_prompt")
        if use_custom:
            new_prompt_name = st.text_input("New Prompt Name (Batch)", value="", key="batch_new_prompt_name")
            custom_template = st.text_area("Custom Prompt Template (Batch)", value=DEFAULT_SYSTEM_PROMPT, height=200, key="batch_custom_prompt_text")
            template_text = custom_template
            prompt_id = None
            if st.button("Save Custom Prompt (Batch)"):
                if new_prompt_name.strip() == "":
                    st.warning("Please enter a name for your custom prompt.")
                elif not id_token or not user_id or not user_name:
                    st.warning("You must be logged in to save prompts.")
                else:
                    ok, prompt_id = save_prompt_firestore(id_token, user_id, user_name, new_prompt_name, custom_template)
                    if ok:
                        st.success(f"Custom prompt '{new_prompt_name}' saved!")
                        st.session_state['batch_custom_prompt_id'] = prompt_id
                        st.rerun()
                    else:
                        st.error("Failed to save prompt to Firestore.")
        # When running batch, use the saved prompt_id if custom
        if use_custom and 'batch_custom_prompt_id' in st.session_state:
            prompt_id = st.session_state['batch_custom_prompt_id']
        st.text_area("Prompt Template (Read Only, Batch)", template_text, height=200, disabled=True, key="batch_prompt_text_readonly")
        st.markdown("---")
        st.subheader("Batch Test All Persona-Question Combos")
        if st.button("Test Prompt for All Personas & Questions"):
            # 1. Determine next batch name
            FIRESTORE_ROOT = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"
            BATCHES_URL = f"{FIRESTORE_ROOT}/prompt_batch_evaluator"
            headers = {"Authorization": f"Bearer {id_token}"}
            resp = session.get(BATCHES_URL, headers=headers)
            existing_batches = [doc['name'].split('/')[-1] for doc in resp.json().get('documents', []) if 'name' in doc]
            next_batch_num = 1
            while f"batch_{next_batch_num}" in existing_batches:
                next_batch_num += 1
            batch_name = f"batch_{next_batch_num}"

            # 3. Generate all queries
            all_queries = []
            for persona in personas:
                for q in questions:
                    query = f"I'm a {persona} {q['question']}"
                    all_queries.append({
                        'persona': persona,
                        'category': q['category'],
                        'question': q['question'],
                        'query': query
                    })
            st.write(f"Total queries to run: {len(all_queries)}")

            # 2. Create batch metadata (minimal fields) AFTER all_queries is built
            metadata_doc = {
                "batch_name": batch_name,
                "created_at": datetime.utcnow().isoformat() + 'Z',
                "prompt_id": prompt_id or '',
                "prompt_text": template_text,
                "num_questions": len(all_queries)
            }
            metadata_url = f"{FIRESTORE_ROOT}/prompt_batch_evaluator/{batch_name}"
            session.patch(metadata_url, headers=headers, json={"fields": {k: {"stringValue": str(v)} for k, v in metadata_doc.items()}})

            st.session_state['batch_eval_results'] = []
            progress = st.progress(0, text="Running batch queries...")
            for i, qd in enumerate(all_queries):
                try:
                    import asyncio
                    st.write(f"Running query {i+1}/{len(all_queries)}: {qd['query']}")
                    result = asyncio.run(qa_service.process_query(
                        query=qd['query'],
                        prompt_template=template_text
                    ))
                    quality_check = result.get('metadata', {}).get('quality_check', {})
                except Exception as e:
                    st.error(f"Error running query {i+1}: {e}")
                    result = {'answer': f"Error: {e}", 'sources': [], 'metadata': {}}
                    quality_check = {}
                # 4. Write each result to the batch's 'results' subcollection
                # Only keep selected keys for quality_check
                qc = result.get('metadata', {}).get('quality_check', {})
                quality_check_filtered = {
                    'confidence_score': qc.get('confidence_score'),
                    'missing_elements': qc.get('missing_elements'),
                    'reasoning': qc.get('reasoning')
                }
                result_doc = {
                    "batch_name": batch_name,
                    "prompt_id": prompt_id or '',
                    "prompt_text": template_text,
                    "query": qd['query'],
                    "question": qd['question'],
                    "category": qd['category'],
                    "persona": qd['persona'],
                    "response": result.get('answer', ''),
                    "sources": json.dumps(result.get('sources', [])),
                    "quality_check": json.dumps(quality_check_filtered),
                    "evaluations": json.dumps([]),
                    "created_at": datetime.utcnow().isoformat() + 'Z'
                }
                result_url = f"{FIRESTORE_ROOT}/prompt_batch_evaluator/{batch_name}/results"
                resp = session.post(result_url, headers=headers, json={"fields": {k: {"stringValue": v} for k, v in result_doc.items()}})
                if resp.status_code in (200, 201):
                    st.write(f"Saved batch result for {qd['persona']} | {qd['question']}")
                else:
                    st.error(f"Failed to save batch result: {resp.text}")
                    logger.error(f"Failed to save batch result for {qd['persona']} | {qd['question']}: {resp.text}")
                st.session_state['batch_eval_results'].append(result_doc)
                progress.progress((i+1)/len(all_queries), text=f"Completed {i+1}/{len(all_queries)}")
            progress.empty()
            st.success("Batch queries completed!")
        # Always display results if available
        if st.session_state.get('batch_eval_results'):
            st.write(f"Displaying {len(st.session_state['batch_eval_results'])} results...")
            st.subheader("Batch Evaluation Results")
            # Group by persona, then by category
            persona_groups = defaultdict(list)
            for item in st.session_state['batch_eval_results']:
                persona_groups[item['persona']].append(item)
            for persona, items in persona_groups.items():
                st.markdown(f"## Persona: {persona}")
                cat_groups = defaultdict(list)
                for item in items:
                    cat_groups[item['category']].append(item)
                for cat, cat_items in cat_groups.items():
                    st.markdown(f"### Category: {cat}")
                    for idx, item in enumerate(cat_items):
                        st.markdown(f"#### Q{idx+1}: {item['question']}")
                        st.markdown(f"**Query:** {item['query']}")
                        st.markdown(f"**Response:** {item['response']}")
                        # Show all existing evaluations
                        evals = []
                        try:
                            evals = json.loads(item.get('evaluations', '[]'))
                        except Exception:
                            evals = []
                        if evals:
                            st.markdown("**All Evaluations:**")
                            for eidx, e in enumerate(evals):
                                st.markdown(f"- **{e.get('evaluator_name', '')}** | Rating: {e.get('rating', '')}/10 | {e.get('timestamp', '')}")
                                if e.get('feedback', ''):
                                    st.markdown(f"    - Feedback: {e['feedback']}")
                        # Allow current user to add a new evaluation
                        st.markdown("**Add Your Evaluation:**")
                        rating = st.slider(f"Rating (1-10) for Q{idx+1} ({persona}, {cat})", 1, 10, 5, key=f"batch_rating_{persona}_{cat}_{idx}")
                        feedback = st.text_area(f"Feedback for Q{idx+1} ({persona}, {cat})", key=f"batch_feedback_{persona}_{cat}_{idx}")
                        if st.button(f"Save Evaluation for Q{idx+1} ({persona}, {cat})", key=f"batch_save_eval_{persona}_{cat}_{idx}"):
                            # Append evaluation to Firestore document
                            eval_data = {
                                'evaluator_name': user_name,
                                'user_id': user_id,
                                'rating': rating,
                                'timestamp': datetime.utcnow().isoformat() + 'Z',
                                'feedback': feedback
                            }
                            # Fetch existing evaluations
                            FIRESTORE_BATCH_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/prompt_batch_evaluator/{item['doc_id']}"
                            headers = {"Authorization": f"Bearer {id_token}"}
                            get_resp = session.get(FIRESTORE_BATCH_URL, headers=headers)
                            if get_resp.status_code == 200:
                                fields = get_resp.json().get('fields', {})
                                evals_json = fields.get('evaluations', {}).get('stringValue', '[]')
                                try:
                                    evals = json.loads(evals_json)
                                except Exception:
                                    evals = []
                                evals.append(eval_data)
                                patch_doc = {"fields": {"evaluations": {"stringValue": json.dumps(evals)}}}
                                patch_resp = session.patch(FIRESTORE_BATCH_URL, headers=headers, json=patch_doc)
                                if patch_resp.status_code in (200, 201):
                                    st.success("Evaluation appended to batch collection!")
                                else:
                                    st.error(f"Failed to append evaluation: {patch_resp.text}")
                            else:
                                st.error(f"Failed to fetch batch doc for evaluation: {get_resp.text}")
                        st.markdown("---")

    # --- View Batch Results Tab ---
    with tabs[3]:
        st.subheader("View Batch Results")
        id_token = st.session_state.get("id_token")
        user_id = st.session_state.get("user_id")
        user_name = st.session_state.get("evaluator_name")
        
        if not id_token:
            st.warning("Please log in to view batch results.")
            return

        # Fetch all batch results from Firestore
        FIRESTORE_BATCH_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/prompt_batch_evaluator"
        headers = {"Authorization": f"Bearer {id_token}"}
        
        # Function to fetch all documents with pagination
        def fetch_all_documents(url):
            all_docs = []
            next_page_token = None
            
            while True:
                # Add page token if we have one
                current_url = f"{url}?pageSize=100"
                if next_page_token:
                    current_url += f"&pageToken={next_page_token}"
                
                resp = session.get(current_url, headers=headers)
                if resp.status_code != 200:
                    st.error(f"Failed to fetch documents: {resp.text}")
                    break
                
                data = resp.json()
                docs = data.get("documents", [])
                all_docs.extend(docs)
                
                # Check if there are more pages
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break
            
            return all_docs
        
        # Fetch all batch documents
        batch_docs = fetch_all_documents(FIRESTORE_BATCH_URL)
        all_results = []
        
        # Process batch documents and their results
        for batch_doc in batch_docs:
            batch_name = batch_doc.get("name", "").split("/")[-1]
            results_url = f"{FIRESTORE_BATCH_URL}/{batch_name}/results"
            
            # Fetch all results for this batch with pagination
            results_docs = fetch_all_documents(results_url)
            
            for doc in results_docs:
                fields = doc.get("fields", {})
                prompt_text = batch_doc.get("fields", {}).get("prompt_text", {}).get("stringValue", "Unknown Prompt")
                prompt_id = batch_doc.get("fields", {}).get("prompt_id", {}).get("stringValue", "")
                doc_id = doc.get("name", "").split("/")[-1]
                
                # Parse evaluations to determine status
                evals = []
                try:
                    evals = json.loads(fields.get("evaluations", {}).get("stringValue", "[]"))
                except Exception:
                    evals = []
                
                item = {
                    'doc_id': doc_id,
                    'batch_name': batch_name,
                    'prompt_id': prompt_id,
                    'prompt_text': prompt_text,
                    'persona': fields.get("persona", {}).get("stringValue", ""),
                    'category': fields.get("category", {}).get("stringValue", ""),
                    'question': fields.get("question", {}).get("stringValue", ""),
                    'query': fields.get("query", {}).get("stringValue", ""),
                    'response': fields.get("response", {}).get("stringValue", ""),
                    'sources': fields.get("sources", {}).get("stringValue", ""),
                    'quality_check': fields.get("quality_check", {}).get("stringValue", ""),
                    'evaluations': evals,
                    'status': "✅" if evals else "❌"
                }
                all_results.append(item)

        # Group results by batch
        batch_groups = defaultdict(list)
        for item in all_results:
            batch_groups[item['batch_name']].append(item)

        # Display batch selector
        selected_batch = st.selectbox(
            "Select Batch",
            options=list(batch_groups.keys()),
            format_func=lambda x: f"Batch: {x}"
        )

        if selected_batch:
            batch_items = batch_groups[selected_batch]
            prompt_text = batch_items[0]['prompt_text'] if batch_items else "No prompt text available"
            
            # Batch header with collapsible prompt
            with st.expander("📋 Batch Details"):
                st.markdown(f"**Batch Name:** {selected_batch}")
                st.markdown(f"**Prompt:** {prompt_text}")
            
            # Add filters
            st.markdown("### Filters")
            col1, col2 = st.columns(2)
            
            # Get unique categories and personas
            categories = sorted(set(item['category'] for item in batch_items))
            personas = sorted(set(item['persona'] for item in batch_items))
            
            with col1:
                selected_category = st.selectbox(
                    "Filter by Category",
                    options=["All"] + categories
                )
            with col2:
                selected_persona = st.selectbox(
                    "Filter by Persona",
                    options=["All"] + personas
                )
            
            # Filter items based on selection
            filtered_items = batch_items
            if selected_category != "All":
                filtered_items = [item for item in filtered_items if item['category'] == selected_category]
            if selected_persona != "All":
                filtered_items = [item for item in filtered_items if item['persona'] == selected_persona]
            
            # Sort items by category and persona
            filtered_items.sort(key=lambda x: (x['category'], x['persona']))
            
            # Create table for results
            st.markdown("### Results")
            
            # Table header
            col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 3, 4, 2, 3])
            with col1:
                st.markdown("**#**")
            with col2:
                st.markdown("**Category**")
            with col3:
                st.markdown("**Persona**")
            with col4:
                st.markdown("**Question**")
            with col5:
                st.markdown("**Status**")
            with col6:
                st.markdown("**Actions**")
            
            # Table rows
            for idx, item in enumerate(filtered_items, 1):
                # Determine if the logged-in user has an evaluation for this item
                user_evals = [e for e in item['evaluations'] if e.get('user_id') == user_id]
                user_has_eval = bool(user_evals)
                status_icon = "✅" if user_has_eval else "❌"
                
                col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 3, 4, 2, 3])
                with col1:
                    st.markdown(str(idx))
                with col2:
                    st.markdown(item['category'])
                with col3:
                    st.markdown(item['persona'])
                with col4:
                    st.markdown(item['question'])
                with col5:
                    st.markdown(f'<span style="color: {"#00cc44" if user_has_eval else "#ff3333"}; font-size: 1.5em;">{status_icon}</span>', unsafe_allow_html=True)
                with col6:
                    view_col, delete_col = st.columns([2, 1])
                    with view_col:
                        if st.button("View", key=f"view_{item['doc_id']}"):
                            st.session_state[f"viewing_{item['doc_id']}"] = True
                    with delete_col:
                        if st.button("🗑", key=f"delete_{item['doc_id']}"):
                            if not user_has_eval:
                                st.warning("You must first add your evaluation before you can delete it.")
                            else:
                                FIRESTORE_BATCH_RESULT_DOC_URL = f"{FIRESTORE_BATCH_URL}/{item['batch_name']}/results/{item['doc_id']}"
                                get_resp = session.get(FIRESTORE_BATCH_RESULT_DOC_URL, headers=headers)
                                if get_resp.status_code == 200:
                                    fields = get_resp.json().get('fields', {})
                                    # Extract actual string values for all fields
                                    category = fields.get("category", {}).get("stringValue", "")
                                    persona = fields.get("persona", {}).get("stringValue", "")
                                    question = fields.get("question", {}).get("stringValue", "")
                                    query = fields.get("query", {}).get("stringValue", "")
                                    response = fields.get("response", {}).get("stringValue", "")
                                    # Extract quality_check if present, else empty dict
                                    quality_check_raw = fields.get("quality_check", {}).get("stringValue", "")
                                    if not quality_check_raw:
                                        quality_check = "{}"
                                    else:
                                        try:
                                            # Ensure it's valid JSON
                                            json.loads(quality_check_raw)
                                            quality_check = quality_check_raw
                                        except Exception:
                                            quality_check = json.dumps({})
                                    evals = []
                                    try:
                                        evals = json.loads(fields.get("evaluations", {}).get("stringValue", "[]"))
                                    except Exception:
                                        evals = []
                                    # Remove evaluations for the logged-in user
                                    new_evals = [e for e in evals if e.get('user_id') != user_id]
                                    # Always patch all fields with actual values, including quality_check
                                    patch_doc = {
                                        "fields": {
                                            "category": {"stringValue": category},
                                            "persona": {"stringValue": persona},
                                            "question": {"stringValue": question},
                                            "query": {"stringValue": query},
                                            "response": {"stringValue": response},
                                            "quality_check": {"stringValue": quality_check},
                                            "evaluations": {"stringValue": json.dumps(new_evals)}
                                        }
                                    }
                                    patch_resp = session.patch(FIRESTORE_BATCH_RESULT_DOC_URL, headers=headers, json=patch_doc)
                                    if patch_resp.status_code in (200, 201):
                                        st.success("Your evaluation was deleted!")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete your evaluation: {patch_resp.text}")
                                elif get_resp.status_code == 404:
                                    st.warning("This result no longer exists. It may have been deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to fetch result for evaluation deletion: {get_resp.text}")
                
                # Show result details if viewing
                if st.session_state.get(f"viewing_{item['doc_id']}", False):
                    # Always fetch the latest data for this item from Firestore
                    FIRESTORE_BATCH_RESULT_DOC_URL = f"{FIRESTORE_BATCH_URL}/{item['batch_name']}/results/{item['doc_id']}"
                    get_resp = session.get(FIRESTORE_BATCH_RESULT_DOC_URL, headers=headers)
                    if get_resp.status_code == 200:
                        fields = get_resp.json().get('fields', {})
                        # Use the latest values from Firestore
                        category = fields.get("category", {}).get("stringValue", "")
                        persona = fields.get("persona", {}).get("stringValue", "")
                        question = fields.get("question", {}).get("stringValue", "")
                        query = fields.get("query", {}).get("stringValue", "")
                        response = fields.get("response", {}).get("stringValue", "")
                        # Parse evaluations
                        try:
                            evals = json.loads(fields.get("evaluations", {}).get("stringValue", "[]"))
                        except Exception:
                            evals = []
                        user_evals = [e for e in evals if e.get('user_id') == user_id]
                    else:
                        # Fallback to item if fetch fails
                        category = item['category']
                        persona = item['persona']
                        question = item['question']
                        response = item['response']
                        user_evals = [e for e in item['evaluations'] if e.get('user_id') == user_id]
                    
                    st.markdown("---")
                    st.markdown("### 📄 Batch Result Details")
                    st.markdown(f"**Category:** {category}")
                    st.markdown(f"**Persona:** {persona}")
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Query:** {query}")
                    st.markdown(f"**Response:** {response}")
                    
                    # Show only the logged-in user's evaluation(s)
                    if user_evals:
                        st.markdown("**Your Evaluation:**")
                        for eval_data in user_evals:
                            st.markdown(f"- **{eval_data.get('evaluator_name', '')}** | Rating: {eval_data.get('rating', '')}/10 | {eval_data.get('timestamp', '')}")
                            if eval_data.get('feedback', ''):
                                st.markdown(f"    - Feedback: {eval_data['feedback']}")
                    else:
                        st.info("You have not submitted an evaluation for this question.")
                    
                    # Add new evaluation (if not already present)
                    if not user_evals:
                        st.markdown("**Add Your Evaluation:**")
                        rating = st.slider("Rating", 1, 10, 5, key=f"rating_{item['doc_id']}")
                        feedback = st.text_area("Feedback", key=f"feedback_{item['doc_id']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save Evaluation", key=f"save_eval_{item['doc_id']}"):
                                eval_data = {
                                    'evaluator_name': user_name,
                                    'user_id': user_id,
                                    'rating': rating,
                                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                                    'feedback': feedback
                                }
                                FIRESTORE_BATCH_RESULT_DOC_URL = f"{FIRESTORE_BATCH_URL}/{item['batch_name']}/results/{item['doc_id']}"
                                get_resp = session.get(FIRESTORE_BATCH_RESULT_DOC_URL, headers=headers)
                                if get_resp.status_code == 200:
                                    fields = get_resp.json().get('fields', {})
                                    # Extract actual string values for all fields
                                    category = fields.get("category", {}).get("stringValue", "")
                                    persona = fields.get("persona", {}).get("stringValue", "")
                                    question = fields.get("question", {}).get("stringValue", "")
                                    query = fields.get("query", {}).get("stringValue", "")
                                    response = fields.get("response", {}).get("stringValue", "")
                                    # Extract quality_check if present, else empty dict
                                    quality_check_raw = fields.get("quality_check", {}).get("stringValue", "")
                                    if not quality_check_raw:
                                        quality_check = "{}"
                                    else:
                                        try:
                                            # Ensure it's valid JSON
                                            json.loads(quality_check_raw)
                                            quality_check = quality_check_raw
                                        except Exception:
                                            quality_check = json.dumps({})
                                    evals = []
                                    try:
                                        evals = json.loads(fields.get("evaluations", {}).get("stringValue", "[]"))
                                    except Exception:
                                        evals = []
                                    evals.append(eval_data)
                                    # Always patch all fields with actual values, including quality_check
                                    patch_doc = {
                                        "fields": {
                                            "category": {"stringValue": category},
                                            "persona": {"stringValue": persona},
                                            "question": {"stringValue": question},
                                            "query": {"stringValue": query},
                                            "response": {"stringValue": response},
                                            "quality_check": {"stringValue": quality_check},
                                            "evaluations": {"stringValue": json.dumps(evals)}
                                        }
                                    }
                                    patch_resp = session.patch(FIRESTORE_BATCH_RESULT_DOC_URL, headers=headers, json=patch_doc)
                                    if patch_resp.status_code in (200, 201):
                                        st.success("Evaluation saved!")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to save evaluation: {patch_resp.text}")
                                elif get_resp.status_code == 404:
                                    st.warning("This result no longer exists. It may have been deleted.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to fetch result for evaluation: {get_resp.text}")
                        with col2:
                            if st.button("Close", key=f"close_{item['doc_id']}"):
                                st.session_state[f"viewing_{item['doc_id']}"] = False
                                st.rerun()
                    else:
                        if st.button("Close", key=f"close_{item['doc_id']}"):
                            st.session_state[f"viewing_{item['doc_id']}"] = False
                            st.rerun()
            
            # Batch management
            st.markdown("---")
            st.markdown("### Batch Management")
            delete_all = st.button("Delete All Results for This Batch")
            confirm_delete = st.checkbox("Confirm delete all results")
            
            if delete_all and confirm_delete:
                for item in batch_items:
                    FIRESTORE_BATCH_DOC_URL = f"{FIRESTORE_BATCH_URL}/{item['doc_id']}"
                    del_resp = session.delete(FIRESTORE_BATCH_DOC_URL, headers=headers)
                st.success("All results for this batch deleted!")
                st.rerun()
            elif delete_all and not confirm_delete:
                st.warning("Please check the confirmation box to delete all results for this batch.")

if __name__ == "__main__":
    prompt_evaluator_ui()
