import os
import logging
import PyPDF2
import streamlit as st
# Using older OpenAI API style for compatibility with version 0.28.1
import openai
from typing import List, Dict, Any, Tuple, Optional

from config import OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperManager:
    """Class to manage paper processing and metadata extraction"""
    
    def __init__(self):
        """Initialize the paper manager"""
        self.vectorstore = None
        if 'chroma_instance' in st.session_state:
            self.vectorstore = st.session_state.chroma_instance
    
    def clean_title(self, title: str) -> str:
        """Clean and standardize paper title format."""
        # Remove any extra whitespace
        cleaned = " ".join(title.split())
        
        # Remove common prefixes like "Title:" or "Paper Title:"
        prefixes = ["title:", "paper title:", "research title:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes if they wrap the entire title
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        
        return cleaned
    
    def extract_first_page_text(self, file_path: str) -> Optional[str]:
        """Extract text from first page of PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    return pdf_reader.pages[0].extract_text()
                return None
        except Exception as e:
            logger.error(f"Error extracting first page from {file_path}: {str(e)}")
            return None
    
    def get_paper_info(self) -> Tuple[int, List[str]]:
        """Get paper counts and titles with improved reliability"""
        try:
            if not self.vectorstore:
                if 'chroma_instance' in st.session_state:
                    self.vectorstore = st.session_state.chroma_instance
                else:
                    return 0, []
            
            # Get all documents
            all_docs = self.vectorstore.get()
            
            # Extract unique document sources
            unique_sources = set()
            for metadata in all_docs.get('metadatas', []):
                if metadata and 'source' in metadata:
                    unique_sources.add(metadata['source'])
            
            return len(unique_sources), list(unique_sources)
        except Exception as e:
            logger.error(f"Error getting paper info: {str(e)}")
            return 0, []
    
    def get_full_text(self, filename: str) -> Optional[str]:
        """Get full text of a paper"""
        try:
            file_path = os.path.join("./results", filename)
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting full text from {filename}: {str(e)}")
            return None
    
    def get_title_from_llm(self, first_page_text: str) -> Optional[str]:
        """Extract and clean paper title using LLM"""
        try:
            prompt = f"""
            Extract the exact title of this research paper from its first page.
            Return only the title without quotes or additional text.
            Return "Unknown Title" if unclear.

            First Page Content:
            {first_page_text[:2000]}
            """
            
            # Use direct OpenAI API approach with older API style for version 0.28.1
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            title = response['choices'][0]['message']['content'].strip()
            if title == "Unknown Title":
                return None
                
            # Clean and standardize the title
            cleaned_title = self.clean_title(title)
            logger.info(f"Extracted and cleaned title: {cleaned_title}")
            return cleaned_title
        except Exception as e:
            logger.error(f"Error extracting title with LLM: {str(e)}")
            return None
    
    def process_paper(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Process a single paper and return (title, error)"""
        try:
            # Extract first page text
            first_page_text = self.extract_first_page_text(file_path)
            if not first_page_text:
                return None, "Failed to extract text from first page"
            
            # Extract title using LLM
            title = self.get_title_from_llm(first_page_text)
            if not title:
                # Fallback to filename as title
                filename = os.path.basename(file_path)
                title = os.path.splitext(filename)[0]
                return title, "Used filename as title (LLM extraction failed)"
            
            return title, None
        except Exception as e:
            logger.error(f"Error processing paper {file_path}: {str(e)}")
            return None, str(e)
    
    def sync_papers(self) -> Tuple[int, int, List[str]]:
        """Sync papers to vector store, checking for existing documents first"""
        try:
            results_dir = "./results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                return 0, 0, ["Created 'results' directory. Please add PDF documents to it."]
            
            # Get all PDF files from the results directory
            pdf_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.pdf')]
            if not pdf_files:
                return 0, 0, ["No PDF files found in the 'results' directory."]
            
            # Get existing paper titles
            _, existing_titles = self.get_paper_info()
            
            # Process each PDF file
            new_papers = []
            errors = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(results_dir, pdf_file)
                
                # Check if this paper is already processed
                if pdf_file in existing_titles:
                    continue
                
                # Process the paper
                title, error = self.process_paper(pdf_path)
                if title:
                    new_papers.append({
                        'name': pdf_file,
                        'title': title,
                        'path': pdf_path
                    })
                if error:
                    errors.append(f"Error processing {pdf_file}: {error}")
            
            return len(pdf_files), len(new_papers), errors
        except Exception as e:
            logger.error(f"Error syncing papers: {str(e)}")
            return 0, 0, [f"Error syncing papers: {str(e)}"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

# Function to process documents
def process_documents(check_for_new=False):
    """Process documents and return document info"""
    results_dir = "./results"
    documents = []
    new_documents = []
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        st.warning("Created 'results' directory. Please add PDF documents to it.")
        return [], []
    
    # Get all PDF files from the results directory
    pdf_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in the 'results' directory.")
        return [], []
    
    # Get list of already processed documents
    processed_doc_names = [doc["name"] for doc in st.session_state.documents] if check_for_new else []
    
    # Create a progress bar
    total_files = len(pdf_files)
    progress_bar = st.progress(0)
    st.write(f"Processing {total_files} documents...")
    
    # Create a container for the file status
    status_container = st.empty()
    
    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files):
        # Update progress bar
        progress_percent = (i) / total_files
        progress_bar.progress(progress_percent)
        
        # Display current file being processed using a slider
        with status_container.container():
            st.slider(
                "Current file", 
                min_value=1, 
                max_value=total_files, 
                value=i+1, 
                disabled=True,
                label_visibility="visible"
            )
            st.info(f"Processing: {pdf_file}")
        
        pdf_path = os.path.join(results_dir, pdf_file)
        try:
            # Check if this is a new document
            is_new = pdf_file not in processed_doc_names
            
            text = extract_text_from_pdf(pdf_path)
            doc_info = {"name": pdf_file, "content": text, "path": pdf_path}
            documents.append(doc_info)
            
            if is_new and check_for_new:
                new_documents.append(doc_info)
                
        except Exception as e:
            st.error(f"Error processing {pdf_file}: {str(e)}")
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text = st.empty()
    status_text.write(f"✅ Completed processing {total_files} documents!")
    
    return documents, new_documents
