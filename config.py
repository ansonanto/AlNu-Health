import os
import streamlit as st
from pathlib import Path

# Get API keys from secrets
OPENAI_API_KEY = st.secrets["api_keys"]["openai"]
GEMINI_API_KEY = st.secrets["api_keys"]["gemini"]
SEMANTIC_SCHOLAR_API_KEY = st.secrets["api_keys"]["semantic_scholar"]

# Get Firebase configuration from secrets
FIREBASE_CONFIG = {
    "apiKey": st.secrets["firebase"]["apiKey"],
    "authDomain": st.secrets["firebase"]["authDomain"],
    "projectId": st.secrets["firebase"]["projectId"],
    "storageBucket": st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId": st.secrets["firebase"]["appId"],
    "measurementId": st.secrets["firebase"]["measurementId"]
}

# Get Google OAuth credentials from secrets
GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["client_id"]
GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]

# Get model settings from secrets
DEFAULT_MODEL = st.secrets["settings"]["default_model"]
EMBEDDING_MODEL = st.secrets["settings"]["embedding_model"]

# Get USDA API key from secrets
USDA_API_KEY = st.secrets["usda"]["api_key"]

# Get email from secrets
EMAIL_ID = st.secrets["credentials"]["email"]

# Vector store path
VECTOR_STORE_PATH = "./simple_vector_storage"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Database path
DB_PATH = Path(st.secrets["database"]["path"])
DB_PATH.mkdir(parents=True, exist_ok=True)

# Create necessary directories
os.makedirs("results", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths and settings
VECTOR_STORE_PATH = "./simple_vector_storage"
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "embedding-001"

# Gemini model configuration
GEMINI_CONFIG = {
    "model_name": "gemini-2.0-flash",
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 2048
}

# Embedding configuration
# Note: This is just a starting value - the actual dimension will be determined
# at runtime based on what the Gemini API returns. The FAISSVectorStore will
# adapt to the actual dimension and persist it for consistency.
EMBEDDING_DIMENSION = 768  # Default dimension for Gemini embeddings

# Load configuration from Streamlit secrets if available
try:
    import streamlit as st
    
    # Load API keys if not already set by environment variables
    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        if 'api_keys' in st.secrets and 'openai' in st.secrets.api_keys:
            secret_key = st.secrets.api_keys.openai
            if secret_key != "your-openai-api-key":
                OPENAI_API_KEY = secret_key
                
    if not SEMANTIC_SCHOLAR_API_KEY or SEMANTIC_SCHOLAR_API_KEY == "":
        if 'api_keys' in st.secrets and 'semantic_scholar' in st.secrets.api_keys:
            secret_scholar_key = st.secrets.api_keys.semantic_scholar
            if secret_scholar_key != "your-semantic-scholar-api-key":
                SEMANTIC_SCHOLAR_API_KEY = secret_scholar_key
                
    if not GEMINI_API_KEY or GEMINI_API_KEY == "":
        if 'api_keys' in st.secrets and 'gemini' in st.secrets.api_keys:
            secret_gemini_key = st.secrets.api_keys.gemini
            if secret_gemini_key != "your-gemini-api-key":
                GEMINI_API_KEY = secret_gemini_key
    
    # Load Firebase configuration
    if 'firebase' in st.secrets:
        try:
            # Try to access as nested attributes
            FIREBASE_CONFIG["apiKey"] = st.secrets.firebase.apiKey
            FIREBASE_CONFIG["authDomain"] = st.secrets.firebase.authDomain
            if "databaseURL" in st.secrets.firebase:
                FIREBASE_CONFIG["databaseURL"] = st.secrets.firebase.databaseURL
            FIREBASE_CONFIG["projectId"] = st.secrets.firebase.projectId
            FIREBASE_CONFIG["storageBucket"] = st.secrets.firebase.storageBucket
            FIREBASE_CONFIG["messagingSenderId"] = st.secrets.firebase.messagingSenderId
            FIREBASE_CONFIG["appId"] = st.secrets.firebase.appId
            if "measurementId" in st.secrets.firebase:
                FIREBASE_CONFIG["measurementId"] = st.secrets.firebase.measurementId
            # Load Google OAuth credentials
            if "googleClientId" in st.secrets.firebase:
                FIREBASE_CONFIG["googleClientId"] = st.secrets.firebase.googleClientId
            if "googleClientSecret" in st.secrets.firebase:
                FIREBASE_CONFIG["googleClientSecret"] = st.secrets.firebase.googleClientSecret
        except Exception as e:
            # Alternative: try to access as dictionary
            print(f"Trying alternative Firebase config loading: {e}")
            try:
                FIREBASE_CONFIG["apiKey"] = st.secrets["firebase"]["apiKey"]
                FIREBASE_CONFIG["authDomain"] = st.secrets["firebase"]["authDomain"]
                if "databaseURL" in st.secrets["firebase"]:
                    FIREBASE_CONFIG["databaseURL"] = st.secrets["firebase"]["databaseURL"]
                FIREBASE_CONFIG["projectId"] = st.secrets["firebase"]["projectId"]
                FIREBASE_CONFIG["storageBucket"] = st.secrets["firebase"]["storageBucket"]
                FIREBASE_CONFIG["messagingSenderId"] = st.secrets["firebase"]["messagingSenderId"]
                FIREBASE_CONFIG["appId"] = st.secrets["firebase"]["appId"]
                if "measurementId" in st.secrets["firebase"]:
                    FIREBASE_CONFIG["measurementId"] = st.secrets["firebase"]["measurementId"]
                # Load Google OAuth credentials
                if "googleClientId" in st.secrets["firebase"]:
                    FIREBASE_CONFIG["googleClientId"] = st.secrets["firebase"]["googleClientId"]
                if "googleClientSecret" in st.secrets["firebase"]:
                    FIREBASE_CONFIG["googleClientSecret"] = st.secrets["firebase"]["googleClientSecret"]
            except Exception as e2:
                print(f"Failed to load Firebase configuration: {e2}")
        
        # Add a default databaseURL if not provided
        if "databaseURL" not in FIREBASE_CONFIG or not FIREBASE_CONFIG["databaseURL"]:
            if "projectId" in FIREBASE_CONFIG and FIREBASE_CONFIG["projectId"]:
                FIREBASE_CONFIG["databaseURL"] = f"https://{FIREBASE_CONFIG['projectId']}.firebaseio.com"
    
    # Get model settings from secrets if available
    # Note: We're using FAISS instead of ChromaDB for better compatibility with Streamlit
    if 'settings' in st.secrets:
        if 'default_model' in st.secrets.settings:
            MODEL_NAME = st.secrets.settings.default_model
        if 'embedding_model' in st.secrets.settings:
            EMBEDDING_MODEL = st.secrets.settings.embedding_model
    
    print("Using Streamlit secrets for configuration")
    
except Exception as e:
    # Already using environment variables
    print(f"Using environment variables: {str(e)}")
    
    # Default paths and settings
    MODEL_NAME = "gemini-2.0-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"

# Paths
PAPERS_DIR = "./reports"
RESULTS_FOLDER = "./results"
PROMPTS_DIR = "./prompts"
EVALUATIONS_DIR = "./evaluations"
