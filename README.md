# AlNu Health - Medical Research RAG System

A comprehensive system for downloading, managing, and querying medical research papers using Retrieval-Augmented Generation (RAG).

## Features

- **Document Management**: Upload and process PDF research papers
- **PubMed Integration**: Search and download papers directly from PubMed
- **Vector Database**: Store and retrieve papers using semantic search
- **RAG-based Querying**: Ask questions about your research papers
- **Metadata Management**: Track author h-index, journal impact factor, and more
- **Prompt Evaluator**: Test different prompts, evaluate responses, and save scores for comparison

## Deployment Instructions

### Deploying to Streamlit Community Cloud

1. Create a GitHub repository and push this code to it
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and the main file (`app.py`)
6. Click on "Advanced settings" and add your secrets in TOML format:

```toml
# Copy and paste this into the Secrets field in Advanced settings
[api_keys]
openai = "your-openai-api-key"
semantic_scholar = "your-semantic-scholar-api-key"

[credentials]
email = "your-email@example.com"  # Used for PubMed API access

[database]
path = "./chroma_db"

[settings]
debug = false
default_model = "gpt-4o"
embedding_model = "text-embedding-3-small"
```

**Important**: Never commit the `.streamlit/secrets.toml` file to your repository. This file is only for local development.

**Note**: The repository is already structured according to Streamlit Community Cloud requirements:
- The entrypoint file (`app.py`) is in the root directory
- Dependencies are declared in `requirements.txt` in the root directory
- Configuration is in `.streamlit/config.toml`
- Python version is specified in `runtime.txt`

### Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv alnu_env`
3. Activate the environment: 
   - Windows: `alnu_env\Scripts\activate`
   - Mac/Linux: `source alnu_env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   EMAIL_ID=your_email@example.com
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
   ```
6. Run the app: `streamlit run app.py`

## Project Structure

- `app.py`: Main application entry point
- `main_app.py`: Main UI and application flow
- `document_processor.py`: Document processing functionality
- `vector_db.py`: Vector database management
- `query_processor.py`: Query processing and RAG functionality
- `pubmed_downloader.py`: PubMed integration for downloading papers
- `prompt_evaluator.py`: Prompt testing and evaluation functionality
- `embeddings.py`: Custom embeddings implementation
- `utils.py`: Utility functions
- `config.py`: Configuration settings
- `.streamlit/config.toml`: Streamlit configuration
- `requirements.txt`: Application dependencies
- `runtime.txt`: Python version specification

## Requirements

See `requirements.txt` for a full list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
