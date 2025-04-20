import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# Get API keys from environment variables first (highest priority)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
EMAIL_ID = os.getenv("EMAIL_ID")

# Default paths and settings
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# Only use Streamlit secrets if environment variables are not set
# and we're not using placeholder values
try:
    # Check if running on Streamlit Cloud with secrets
    # First check if secrets are available at all
    if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "":
            if 'openai' in st.secrets["api_keys"]:
                secret_key = st.secrets["api_keys"]["openai"]
                if secret_key != "your-openai-api-key":
                    OPENAI_API_KEY = secret_key
                
        if not SEMANTIC_SCHOLAR_API_KEY or SEMANTIC_SCHOLAR_API_KEY == "":
            if 'semantic_scholar' in st.secrets["api_keys"]:
                secret_scholar_key = st.secrets["api_keys"]["semantic_scholar"]
                if secret_scholar_key != "your-semantic-scholar-api-key":
                    SEMANTIC_SCHOLAR_API_KEY = secret_scholar_key
    
    # Check for credentials section
    if hasattr(st, 'secrets') and 'credentials' in st.secrets:
        if not EMAIL_ID or EMAIL_ID == "":
            if 'email' in st.secrets["credentials"]:
                secret_email = st.secrets["credentials"]["email"]
                if secret_email != "your-email@example.com":
                    EMAIL_ID = secret_email
    
    # Get database path and model settings from secrets if available
    if hasattr(st, 'secrets') and 'database' in st.secrets and 'path' in st.secrets["database"]:
        CHROMA_PATH = st.secrets["database"]["path"]
    
    if hasattr(st, 'secrets') and 'settings' in st.secrets:
        if 'default_model' in st.secrets["settings"]:
            MODEL_NAME = st.secrets["settings"]["default_model"]
        if 'embedding_model' in st.secrets["settings"]:
            EMBEDDING_MODEL = st.secrets["settings"]["embedding_model"]
    
    print("Using Streamlit secrets for configuration")
    
except Exception as e:
    # Already using environment variables
    print(f"Using environment variables: {str(e)}")
    
    # Default paths and settings
    CHROMA_PATH = "./chroma_db"
    MODEL_NAME = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-small"

# Paths
PAPERS_DIR = "./reports"
RESULTS_FOLDER = "./results"
PROMPTS_DIR = "./prompts"
EVALUATIONS_DIR = "./evaluations"
