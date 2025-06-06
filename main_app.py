# Must be the first import
import streamlit as st
import requests
from datetime import datetime
import re
import json
import os
from document_processor import extract_text_from_pdf
from datetime import datetime

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="AlnuHealth - RAG Document Search System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library imports
import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio

# Third-party imports
from PIL import Image
import openai
import google.generativeai as genai

# Import configuration
from config import VECTOR_STORE_PATH

# Import utility functions
from document_processor import process_documents, PaperManager
from vector_db import initialize_vector_db, create_vector_db, check_db_status
from query_processor import query_documents
from pubmed_downloader import pubmed_downloader_ui
from prompt_evaluator import prompt_evaluator_ui
from food_analyzer import process_food_image, format_macro_display
from gemini_services import gemini_service
from qa_service import qa_service
from firebase_helpers import save_evaluation_firestore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API keys from secrets
openai.api_key = st.secrets["api_keys"]["openai"]

# Initialize Gemini
try:
    # Initialize Gemini model
    model = genai.GenerativeModel(
        model_name=st.secrets["settings"]["default_model"],
        generation_config={
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 2048
        }
    )
    # Set API key
    model.api_key = st.secrets["api_keys"]["gemini"]
    logger.info("Gemini model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")
    st.error("Failed to initialize Gemini model. Please check your API key and try again.")
    st.stop()

# Get model settings from secrets
DEFAULT_MODEL = st.secrets["settings"]["default_model"]
EMBEDDING_MODEL = st.secrets["settings"]["embedding_model"]

API_KEY = st.secrets["firebase"]["apiKey"]
PROJECT_ID = st.secrets["firebase"]["projectId"]

REGISTER_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
LOGIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
FIRESTORE_USERS_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/users"
FIRESTORE_EVALS_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/evaluations"

def register_user(name, email, password):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    resp = requests.post(REGISTER_URL, json=payload)
    data = resp.json()
    if "error" in data:
        return False, data["error"]["message"]
    id_token = data["idToken"]
    user_id = data["localId"]
    headers = {"Authorization": f"Bearer {id_token}"}
    doc = {
        "fields": {
            "name": {"stringValue": name},
            "email": {"stringValue": email},
            "created_at": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
            "user_id": {"stringValue": user_id}
        }
    }
    # Use PATCH to set the document ID to user_id
    user_doc_url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/users/{user_id}"
    resp2 = requests.patch(user_doc_url, headers=headers, json=doc)
    if resp2.status_code in (200, 201):
        return True, "Registration successful!"
    else:
        return False, f"Registered, but failed to save user info: {resp2.text}"

def login_user(email, password):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    resp = requests.post(LOGIN_URL, json=payload)
    data = resp.json()
    if "error" in data:
        return False, data["error"]["message"], None
    id_token = data["idToken"]
    user_id = data["localId"]
    # Get user name from Firestore
    headers = {"Authorization": f"Bearer {id_token}"}
    r = requests.get(FIRESTORE_USERS_URL, headers=headers)
    name = None
    user_doc_found = False
    if r.status_code == 200:
        docs = r.json().get("documents", [])
        for doc in docs:
            fields = doc.get("fields", {})
            if fields.get("user_id", {}).get("stringValue") == user_id:
                name = fields.get("name", {}).get("stringValue")
                user_doc_found = True
                break
    return True, "Login successful!", {"idToken": id_token, "localId": user_id, "name": name or "", "user_doc_found": user_doc_found}

def save_evaluation_firestore(id_token, user_id, evaluator_name, prompt, query, response, sources, rating, feedback):
    headers = {"Authorization": f"Bearer {id_token}"}
    doc = {
        "fields": {
            "prompt": {"stringValue": prompt},
            "query": {"stringValue": query},
            "response": {"stringValue": response},
            "sources": {"stringValue": str(sources)},
            "rating": {"integerValue": str(rating)},
            "feedback": {"stringValue": feedback},
            "timestamp": {"timestampValue": datetime.utcnow().isoformat() + "Z"},
            "user_id": {"stringValue": user_id},
            "evaluator_name": {"stringValue": evaluator_name}
        }
    }
    resp = requests.post(FIRESTORE_EVALS_URL, headers=headers, json=doc)
    return resp.status_code == 200

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'new_documents' not in st.session_state:
        st.session_state.new_documents = []
    if 'last_processed_time' not in st.session_state:
        st.session_state.last_processed_time = None
    if 'db_status' not in st.session_state:
        st.session_state.db_status = "Not initialized"
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'accuracy_percentage' not in st.session_state:
        st.session_state.accuracy_percentage = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Document Management"
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'firebase_auth' not in st.session_state:
        st.session_state.firebase_auth = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = None

# Initialize session state variables if they don't exist
initialize_session_state()

# Try to initialize the vector database on startup
try:
    # Debug print: list files in vector DB directory
    print("Files in vector DB directory at startup:", os.listdir(VECTOR_STORE_PATH))
    # Check for existing vector store at startup
    vector_store_path = Path(VECTOR_STORE_PATH)
    if vector_store_path.exists() and (vector_store_path / "index.faiss").exists():
        logger.info("Found existing vector database, attempting to load")
        # Initialize the database
        db = initialize_vector_db()
        if db is not None:
            st.session_state.db = db
            st.session_state.vector_store = db  # Store in both places for compatibility
            st.session_state.processed_docs = True
            st.session_state.db_status = "Vector database loaded successfully"
            logger.info("Vector database loaded successfully")
        else:
            st.session_state.processed_docs = False
            st.session_state.db_status = "Failed to load vector database"
            logger.warning("Failed to load vector database")
    else:
        logger.info("No existing vector database found or verification failed")
        st.session_state.processed_docs = False
        st.session_state.db_status = "No vector database found"
except Exception as e:
    logger.error(f"Error initializing vector database on startup: {str(e)}")
    st.session_state.processed_docs = False
    st.session_state.db_status = f"Error: {str(e)}"

# Initialize additional session state variables for search results and UI state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'accuracy_percentage' not in st.session_state:
    st.session_state.accuracy_percentage = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Document Management"
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False

def save_query_history():
    """Save query history to disk"""
    history_path = os.path.join(VECTOR_STORE_PATH, "query_history.json")
    try:
        if 'query_history' in st.session_state:
            with open(history_path, 'w') as f:
                json.dump(st.session_state.query_history, f)
            logger.info("Query history saved successfully")
    except Exception as e:
        logger.error(f"Error saving query history: {str(e)}")

def load_query_history():
    """Load query history from disk"""
    history_path = os.path.join(VECTOR_STORE_PATH, "query_history.json")
    try:
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            st.session_state.query_history = history
            logger.info("Query history loaded successfully")
    except Exception as e:
        logger.error(f"Error loading query history: {str(e)}")
        st.session_state.query_history = []

def show_auth():
    st.sidebar.title("User Authentication")
    menu = st.sidebar.radio("Menu", ["Login", "Register"])
    if menu == "Register":
        st.sidebar.subheader("Register")
        name = st.sidebar.text_input("Name", key="reg_name")
        email = st.sidebar.text_input("Email", key="reg_email")
        password = st.sidebar.text_input("Password", type="password", key="reg_password")
        if st.sidebar.button("Register"):
            if not name or not email or not password:
                st.sidebar.error("Please fill all fields.")
            else:
                success, msg = register_user(name, email, password)
                if success:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(msg)
    elif menu == "Login":
        st.sidebar.subheader("Login")
        email = st.sidebar.text_input("Email", key="login_email")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            if not email or not password:
                st.sidebar.error("Please fill all fields.")
            else:
                success, msg, data = login_user(email, password)
                if success:
                    st.sidebar.success(msg)
                    st.session_state["id_token"] = data["idToken"]
                    st.session_state["user_id"] = data["localId"]
                    st.session_state["evaluator_name"] = data["name"]
                    st.session_state["email"] = email
                    st.session_state["logged_in"] = True
                else:
                    st.sidebar.error(msg)
    if st.session_state.get("logged_in"):
        user_doc_found = st.session_state.get("user_doc_found", True)
        st.sidebar.success(f"Logged in as {st.session_state['evaluator_name']} ({st.session_state['email']})")
        if not user_doc_found:
            st.sidebar.warning("Your profile is missing. Please re-register or contact support.")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

def main():
    show_auth()
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to use the app.")
        st.stop()
    # Initialize session state
    initialize_session_state()
    
    # Load query history
    if 'query_history' not in st.session_state:
        load_query_history()
    
    # Display header
    st.title("🏥 AlnuHealth - Medical Research RAG System")
    
    # Create tabs for different functionalities
    tabs = ["Document Management", "Search & Query", "PubMed Downloader", "Prompt Evaluator"]
    st.session_state.current_tab = st.radio("Select Functionality:", tabs, horizontal=True)
    
    # Display the selected tab
    if st.session_state.current_tab == "Document Management":
        document_management_ui()
    elif st.session_state.current_tab == "Search & Query":
        search_query_ui()
    elif st.session_state.current_tab == "PubMed Downloader":
        pubmed_downloader_ui()
    elif st.session_state.current_tab == "Prompt Evaluator":
        prompt_evaluator_ui()
    
    # Display footer
    st.markdown("---")
    st.markdown("AlnuHealth - Medical Research RAG System 2025")

def load_organized_docs():
    if os.path.exists("organized_docs.json"):
        with open("organized_docs.json", "r") as f:
            return json.load(f)
    return []

def get_all_pdfs_from_results():
    results_dir = "results"
    return [os.path.join(results_dir, fname) for fname in os.listdir(results_dir) if fname.lower().endswith(".pdf")]

# Load organized docs on app start
if "organized_docs" not in st.session_state:
    st.session_state.organized_docs = load_organized_docs()

def process_new_documents():
    pdf_paths = get_all_pdfs_from_results()
    documents = []
    total = len(pdf_paths)
    progress_bar = st.progress(0, text="Starting document processing...")
    status_text = st.empty()
    for i, pdf_path in enumerate(pdf_paths):
        try:
            status_text.info(f"Processing {os.path.basename(pdf_path)} ({i+1}/{total})...")
            text = extract_text_from_pdf(pdf_path)
            doc_info = {
                "name": os.path.basename(pdf_path),
                "content": text,
                "path": pdf_path,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "upload_date": datetime.utcnow().isoformat(),
                    "file_type": "pdf"
                }
            }
            documents.append(doc_info)
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {e}")
        progress_bar.progress((i+1)/total, text=f"Processed {i+1}/{total} documents")
    if documents:
        status_text.info("Vectorizing documents...")
        from vector_db import create_vector_db
        create_vector_db(documents, update_existing=True)
        progress_bar.progress(1.0, text="All documents processed and vectorized!")
        st.success(f"Processed and vectorized {len(documents)} new documents.")
        status_text.empty()
    else:
        progress_bar.progress(1.0, text="No documents were processed.")
        st.warning("No documents were processed.")
        status_text.empty()

def organize_documents_with_gemini():
    api_key = st.secrets["api_keys"]["gemini"]
    model_name = st.secrets["settings"]["default_model"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    # Only organize new PDFs
    organized_docs = st.session_state.organized_docs.copy()
    progress_bar = st.progress(0, text="Organizing documents...")
    total = len(get_all_pdfs_from_results())
    for idx, pdf_path in enumerate(get_all_pdfs_from_results()):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
        prompt = f'''
You are an expert document classifier. Given the following text, classify the document as one of: "research paper", "report", "guide", or "other". Also, suggest a short topic or main subject for the document.

Document text:
"""
{text[:1000]}
"""

Respond in JSON like:
{{"type": "...", "topic": "..."}}
'''
        try:
            response = model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                doc_type = result.get("type", "other")
                topic = result.get("topic", "")
            else:
                doc_type = "other"
                topic = ""
        except Exception as e:
            doc_type = "other"
            topic = ""
            st.warning(f"Failed to classify {os.path.basename(pdf_path)}: {e}")
        organized_docs.append({
            "title": os.path.basename(pdf_path),
            "path": pdf_path,
            "type": doc_type,
            "topic": topic
        })
        progress_bar.progress((idx + 1) / total, text=f"Organizing {os.path.basename(pdf_path)} ({idx + 1}/{total})")
    progress_bar.empty()
    st.session_state.organized_docs = organized_docs
    # Save to JSON
    with open("organized_docs.json", "w") as f:
        json.dump(organized_docs, f, indent=2)
    st.success("Documents organized by type and topic!")

def document_management_ui():
    """UI for document management tab"""
    st.header("Document Management")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Document processing section
        st.subheader("Document Processing")
        
        # Check database status
        check_db_status()
        
        # Display database status
        st.info(f"Vector Database Status: {st.session_state.db_status}")
        
        # Process documents button
        if get_all_pdfs_from_results() and st.button("Process Documents"):
            process_new_documents()
        
        # Reset database button
        if st.button("Reset Database"):
            # Initialize vector store with reset flag
            if initialize_vector_db(reset_db=True):
                # Clear all relevant session state except query history
                st.session_state.db = None
                st.session_state.vector_store = None
                st.session_state.processed_docs = False
                st.session_state.documents = []
                st.session_state.new_documents = []
                st.session_state.last_processed_time = None
                st.session_state.db_status = "Not initialized"
                
                st.success("Vector database reset successfully")
                st.rerun()
            else:
                st.error("Failed to reset vector database")
        
        # Organize documents button
        if st.button("Organize Documents"):
            organize_documents_with_gemini()
        
        # Display document info
        docs_to_show = st.session_state.get("organized_docs", [])
        types = sorted(set(doc.get("type", "other") for doc in docs_to_show))
        selected_type = st.selectbox("Filter by type", ["All"] + types)
        if selected_type != "All":
            docs_to_show = [doc for doc in docs_to_show if doc.get("type") == selected_type]
        if not docs_to_show:
            st.info("No documents to display.")
        else:
            for doc in docs_to_show:
                st.markdown(f"**Title:** {doc.get('title', '')}")
                st.markdown(f"**Type:** {doc.get('type', '')}")
                st.markdown(f"**Topic:** {doc.get('topic', '')}")
                st.markdown("---")
    
    with col2:
        # Document upload section
        st.subheader("Document Upload")
        
        # File uploader for direct upload and vectorization
        uploaded_file = st.file_uploader("Upload a document (PDF, TXT)", type=['pdf', 'txt'])
        if uploaded_file is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if uploaded_file.name.lower().endswith('.pdf'):
                    from document_processor import extract_text_from_pdf
                    text = extract_text_from_pdf(temp_path)
                else:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                doc_info = {
                    "name": uploaded_file.name,
                    "content": text,
                    "path": temp_path,
                    "metadata": {
                        "source": uploaded_file.name,
                        "upload_date": datetime.utcnow().isoformat(),
                        "file_type": uploaded_file.type
                    }
                }
                progress_bar.progress(0.5)
                status_text.write("Processing document...")
                from vector_db import create_vector_db
                vectorstore = create_vector_db([doc_info], update_existing=True)
                if vectorstore:
                    st.success(f"Successfully processed and added '{uploaded_file.name}' to the vector database!")
                    os.remove(temp_path)
                else:
                    st.error("Failed to add document to vector database.")
                progress_bar.progress(1.0)
                status_text.write("✅ Processing complete!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                progress_bar.progress(1.0)
                status_text.write("❌ Error processing document")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Vectorization status checker
        with st.expander("Check Vectorization Status"):
            results_dir = "results"
            pdf_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.pdf')]
            vectorstore = st.session_state.get("vector_store") or st.session_state.get("db")
            vectorized_docs = set()
            if vectorstore is not None:
                if hasattr(vectorstore, '_metadata'):
                    for metadata in vectorstore._metadata:
                        if metadata and 'source' in metadata:
                            vectorized_docs.add(os.path.basename(metadata['source']))
                elif hasattr(vectorstore, 'get'):
                    all_docs = vectorstore.get()
                    if all_docs and 'metadatas' in all_docs:
                        for metadata in all_docs['metadatas']:
                            if metadata and 'source' in metadata:
                                vectorized_docs.add(os.path.basename(metadata['source']))
            not_vectorized = [f for f in pdf_files if f not in vectorized_docs]
            if not_vectorized:
                st.warning(f"These files are not vectorized: {not_vectorized}")
            else:
                st.success("All files in the results folder are vectorized!")
        
        # Display instructions
        st.markdown("""
        ### Instructions
        1. Upload PDF files to the 'results' directory
        2. Click 'Process Documents' to extract text and create embeddings
        3. Use the 'Search & Query' tab to ask questions about the documents
        """)
        # Display last processed time
        if st.session_state.last_processed_time:
            last_processed = time.strftime(
                "%Y-%m-%d %H:%M:%S", 
                time.localtime(st.session_state.last_processed_time)
            )
            st.info(f"Last processed: {last_processed}")

def search_query_ui():
    """Search and query interface"""
    st.header("Search & Query")
    
    # Chat mode toggle
    chat_mode = st.toggle("Chat Mode", value=st.session_state.get("chat_mode", False))
    st.session_state["chat_mode"] = chat_mode
    
    # Warning if no documents processed
    if not st.session_state.get("processed_docs", False):
        st.warning("No documents have been processed yet. Please go to Document Management to process some documents first.")
        return
    
    # Query input
    query = st.text_input("Enter your question:", key="query_input")
    
    if query:
        try:
            with st.spinner("Processing your query..."):
                # Process query with QA service
                response = asyncio.run(qa_service.process_query(
                    query=query,
                    conversation_id=st.session_state.get("current_conversation_id"),
                    max_sources=5
                ))
                # Store results
                st.session_state["last_response"] = response
                st.session_state["current_conversation_id"] = response["conversation_id"]
                # Display response
                st.markdown("### Response")
                st.write(response["answer"])
                # Display confidence and quality metrics
                quality_check = response["metadata"]["quality_check"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{quality_check['confidence_score']:.2%}")
                with col2:
                    st.metric("Completeness", "Complete" if quality_check["is_complete"] else "Incomplete")
                # Display sources
                st.markdown("### Sources")
                sources = response["sources"]
                # Group sources by type
                doc_sources = [s for s in sources if s.get("metadata", {}).get("source_type") == "document"]
                web_sources = [s for s in sources if s.get("metadata", {}).get("source_type") == "web"]
                # Display document sources
                if doc_sources:
                    st.markdown("#### Research Documents")
                    for i, source in enumerate(doc_sources, 1):
                        with st.expander(f"Document {i} (Score: {source['relevance_score']:.2f})"):
                            st.markdown(f"**Source:** {source['source']}")
                            st.markdown(f"**Content:** {source['chunk']}")
                # Display web sources
                if web_sources:
                    st.markdown("#### Web Sources")
                    for i, source in enumerate(web_sources, 1):
                        with st.expander(f"Web Source {i}"):
                            st.markdown(f"**Title:** {source['title']}")
                            st.markdown(f"**URL:** [{source['source']}]({source['source']})")
                            st.markdown(f"**Domain:** {source['metadata']['domain']}")
                            if source.get('chunk'):
                                st.markdown(f"**Snippet:** {source['chunk']}")
                # Display quality check details
                if not quality_check["is_complete"]:
                    st.warning("The answer may be incomplete. Consider the following:")
                    for element in quality_check["missing_elements"]:
                        st.markdown(f"- {element}")
                if quality_check["needs_web_search"]:
                    st.info("Additional information from web search has been included to provide a more complete answer.")
                # Chat history in chat mode
                if chat_mode:
                    st.markdown("### Conversation History")
                    history = qa_service.get_conversation_history(st.session_state["current_conversation_id"])
                    if history and history["exchanges"]:
                        for exchange in history["exchanges"]:
                            st.markdown(f"**User:** {exchange['user']}")
                            st.markdown(f"**Assistant:** {exchange['assistant']}")
                            st.markdown("---")
                        # Clear conversation button
                        if st.button("Clear Conversation"):
                            st.session_state["current_conversation_id"] = None
                            st.session_state["chat_mode"] = False
                            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Error in search_query_ui: {str(e)}")

if __name__ == "__main__":
    main()
