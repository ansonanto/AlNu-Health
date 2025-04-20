import os
import json
import time
import hashlib
import streamlit as st
from Bio import Entrez
import pdfplumber
from tqdm import tqdm
from metapub import PubMedFetcher
import requests
from bs4 import BeautifulSoup
import subprocess
import sys
import re
import urllib.request
from urllib.parse import urljoin

# Import configuration
from config import EMAIL_ID, SEMANTIC_SCHOLAR_API_KEY, RESULTS_FOLDER

# Enhanced Configuration
PUBMED_ERRORS_FILEPATH = "./unfetch_pmids.txt"
DOWNLOADED_PUBMEDS_FILEPATH = "./downloaded_pubmeds.tsv"
METADATA_FILEPATH = "./articles_metadata.json"
OVERALL_DOWNLOAD_STATUS_FILEPATH = "./pubmed_articles_download_status.json"

# PMC Base URLs
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"

# Initialize PubMed fetcher
FETCH = PubMedFetcher()
Entrez.email = EMAIL_ID

def initial_setup():
    """
    Initial Filesystem setup if not already exists.
    """
    # Check if the results folder exists, if not, create it
    if not os.path.exists(RESULTS_FOLDER):
        print(f"Creating folder: {RESULTS_FOLDER}")
        os.makedirs(RESULTS_FOLDER)

    # Create empty error file if it does not exist
    if not os.path.exists(PUBMED_ERRORS_FILEPATH):
        print(f"Creating errors file: {PUBMED_ERRORS_FILEPATH}")
        with open(PUBMED_ERRORS_FILEPATH, 'w') as f:
            f.write('')
            
    # Create downloaded PMIDs file if it does not exist
    if not os.path.exists(DOWNLOADED_PUBMEDS_FILEPATH):
        print(f"Creating downloaded PMIDs file: {DOWNLOADED_PUBMEDS_FILEPATH}")
        with open(DOWNLOADED_PUBMEDS_FILEPATH, 'w') as f:
            f.write('pmid\tpmc_id\ttitle\n')  # Header row
    
    # Create metadata file if it does not exist
    if not os.path.exists(METADATA_FILEPATH):
        print(f"Creating metadata file: {METADATA_FILEPATH}")
        with open(METADATA_FILEPATH, 'w') as f:
            json.dump([], f)
    
    # Create download status file if it does not exist
    if not os.path.exists(OVERALL_DOWNLOAD_STATUS_FILEPATH):
        print(f"Creating download status file: {OVERALL_DOWNLOAD_STATUS_FILEPATH}")
        with open(OVERALL_DOWNLOAD_STATUS_FILEPATH, 'w') as f:
            json.dump({}, f)

def append_to_error_file(pmid):
    """
    Update the status file for the PMID as the error in downloading the article.
    """
    if os.path.exists(PUBMED_ERRORS_FILEPATH):
        with open(PUBMED_ERRORS_FILEPATH, 'r') as f:
            existing_pmids = f.read().splitlines()
    else:
        existing_pmids = []

    if pmid not in existing_pmids:
        with open(PUBMED_ERRORS_FILEPATH, 'a') as f:
            f.write(f"{pmid}\n")
        print(f"PMID {pmid} added to error file.")
    else:
        print(f"PMID {pmid} is already in the error file. Skipping appending.")

def get_pubmed_search_query(search_term, max_results=10, filter_open_access=True):
    """
    Generate a PubMed API search query for a given search term.
    Optionally filter for open access articles.
    """
    try:
        # Add open access filter if requested
        if filter_open_access:
            search_term = f"{search_term} AND (open access[Filter] OR free full text[Filter])"
            
        # Search PubMed for the given term
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        
        if pmids:
            print(f"Found {len(pmids)} articles matching '{search_term}'")
        else:
            print(f"No articles found for '{search_term}'")
            
        return pmids
    except Exception as e:
        print(f"Error searching PubMed for '{search_term}': {str(e)}")
        return []

def is_article_already_downloaded(pmid):
    """
    Check if the article is already downloaded.
    Checks both the original PMID.pdf format and any file that starts with the PMID
    to account for renamed files with titles.
    """
    # Check for the original filename format
    expected_pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
    if os.path.exists(expected_pdf_path):
        return True
    
    # Check for any file that starts with the PMID (for renamed files with titles)
    if os.path.exists(RESULTS_FOLDER):
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.lower().endswith('.pdf') and filename.startswith(f"{pmid}_"):
                return True
    
    # Also check the downloaded_pubmeds.tsv file for this PMID
    if os.path.exists(DOWNLOADED_PUBMEDS_FILEPATH):
        try:
            with open(DOWNLOADED_PUBMEDS_FILEPATH, 'r') as f:
                for line in f:
                    if line.strip().startswith(pmid + '\t'):
                        return True
        except Exception as e:
            print(f"Error checking downloaded_pubmeds.tsv: {e}")
    
    return False


def calculate_md5(file_path):
    """
    Generate the MD5 hash for the PubMed article.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_pdf_page_length(pdf_path):
    """
    Get the number of pages for a given PDF article.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
        return num_pages
    except Exception as e:
        print(f"Error getting PDF page length: {str(e)}")
        return 0


def get_author_h_index(author_name):
    """
    Get the author's H-index using the Semantic Scholar API.
    """
    if not SEMANTIC_SCHOLAR_API_KEY:
        return 0  # Skip if no API key
        
    base_url = "https://api.semanticscholar.org/graph/v1/author/search"
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params = {
        "query": author_name,
        "fields": "name,authorId",
        "limit": 1
    }
    try:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                author_id = data['data'][0].get('authorId')
                author_details = requests.get(
                    f"https://api.semanticscholar.org/graph/v1/author/{author_id}",
                    headers=headers,
                    params={"fields": "hIndex"}
                ).json()
                return author_details.get("hIndex", 0)  # Default to 0 if no H-index found
    except Exception as e:
        print(f"Error fetching H-index for {author_name}: {e}")
    return 0  # Return 0 if there was an error or no data found


def get_journal_impact_factor(journal_name):
    """
    Retrieve the Journal Impact Factor for a given journal.
    """
    if not journal_name:
        return "Journal title not available"

    journal_name_capitalized = journal_name.upper()
    search_url = "https://wos-journal.info/"
    params = {"jsearch": journal_name_capitalized}
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        title_elements = soup.find_all('div', class_='title col-4 col-md-3')
        for title_element in title_elements:
            if "Journal Impact Factor (JIF):" in title_element.text:
                impact_factor_div = title_element.find_next_sibling('div', class_='content col-8 col-md-9')
                if impact_factor_div:
                    impact_factor_text = impact_factor_div.text.strip()
                    return impact_factor_text
        return "Impact Factor not found."
    except Exception as e:
        print(f"Error fetching impact factor: {e}")
        return f"Error fetching impact factor: {e}"

def get_pmc_id_from_pmid(pmid):
    """
    Get the PMC ID for a given PubMed ID using Entrez.
    """
    try:
        # Use Entrez to get the PMC ID from the PMID
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        record = Entrez.read(handle)
        handle.close()
        
        # Extract PMC ID from the record
        if record and record[0].get("LinkSetDb") and record[0]["LinkSetDb"] and record[0]["LinkSetDb"][0].get("Link"):
            pmc_id = record[0]["LinkSetDb"][0]["Link"][0]["Id"]
            return pmc_id
        
        # If no PMC ID found, try to get the article info to check if it's open access
        handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        if record and "PubmedArticle" in record:
            article = record["PubmedArticle"][0]
            if "PubmedData" in article and "PublicationStatus" in article["PubmedData"]:
                status = article["PubmedData"]["PublicationStatus"]
                if "open access" in status.lower() or "free" in status.lower():
                    # For open access articles without a PMC ID, we'll use the PMID
                    return "OA_" + pmid
        
        return None
    except Exception as e:
        print(f"Error getting PMC ID for PMID {pmid}: {str(e)}")
        return None

def download_from_pmc_direct(pmid, pmc_id):
    """
    Download a paper directly from PubMed Central using the PMC ID.
    """
    try:
        # Try to get the PDF URL from the PMC page
        pmc_url = f"{PMC_BASE_URL}{pmc_id}/"
        print(f"Accessing PMC URL: {pmc_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        response = requests.get(pmc_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to access PMC page: Status code {response.status_code}")
            return False
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for PDF link - try multiple patterns
        pdf_link = None
        
        # Method 1: Look for PDF links in the page
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if ('pdf' in href and ('.pdf' in href or '/pdf/' in href)) or 'download' in href:
                pdf_link = a['href']
                print(f"Found PDF link: {pdf_link}")
                break
        
        # Method 2: Look for the specific PMC PDF format
        if not pdf_link:
            pdf_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/{pmc_id}.pdf"
            print(f"Trying standard PMC PDF format: {pdf_link}")
        
        if pdf_link:
            # Make sure it's a full URL
            if not pdf_link.startswith('http'):
                pdf_link = urljoin(pmc_url, pdf_link)
            
            # Download the PDF with proper headers
            pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
            print(f"Downloading PDF to {pdf_path}")
            
            # Use requests instead of urllib for better error handling
            pdf_response = requests.get(pdf_link, headers=headers)
            if pdf_response.status_code != 200:
                print(f"Failed to download PDF: Status code {pdf_response.status_code}")
                return False
                
            # Save the PDF
            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.content)
            
            # Verify the file was downloaded and is a valid PDF
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:  # Basic size check
                # Check if it's a valid PDF
                try:
                    with open(pdf_path, 'rb') as f:
                        header = f.read(4)
                        if header != b'%PDF':
                            print(f"Downloaded file is not a valid PDF for PMID {pmid}")
                            os.remove(pdf_path)
                            return False
                except Exception as e:
                    print(f"Error checking PDF validity for PMID {pmid}: {str(e)}")
                    return False
                    
                # Get article metadata for better filename
                try:
                    article = FETCH.article_by_pmid(pmid)
                    if article and article.title:
                        # Create a safe filename from the title
                        safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '' for c in article.title])
                        safe_title = safe_title[:100]  # Limit length
                        new_filename = f"{pmid}_{safe_title}.pdf"
                        new_path = os.path.join(RESULTS_FOLDER, new_filename)
                        os.rename(pdf_path, new_path)
                        
                        # Record the download in the TSV file
                        with open(DOWNLOADED_PUBMEDS_FILEPATH, 'a') as f:
                            f.write(f"{pmid}\t{pmc_id}\t{article.title}\n")
                        
                        print(f"Successfully downloaded and renamed PDF for PMID {pmid}")
                        return True
                except Exception as e:
                    print(f"Error getting article metadata for PMID {pmid}: {str(e)}")
                    # Even if metadata fails, we still have the PDF
                    return True
            else:
                print(f"Downloaded file is too small or doesn't exist for PMID {pmid}")
        else:
            print(f"No PDF link found for PMID {pmid}")
        return False
    except Exception as e:
        print(f"Error downloading from PMC direct for PMID {pmid}: {str(e)}")
        return False

def download_using_scihub(pmid):
    """
    Try to download a paper from Sci-Hub using the DOI or PMID.
    """
    try:
        # Get the article metadata to find the DOI
        article = FETCH.article_by_pmid(pmid)
        if not article or not article.doi:
            print(f"No DOI found for PMID {pmid}")
            return False
            
        doi = article.doi
        print(f"Found DOI {doi} for PMID {pmid}")
        
        # List of Sci-Hub domains to try
        scihub_domains = [
            "https://sci-hub.se",
            "https://sci-hub.st",
            "https://sci-hub.ru"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        for domain in scihub_domains:
            try:
                url = f"{domain}/{doi}"
                print(f"Trying Sci-Hub URL: {url}")
                
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for the PDF iframe or download button
                iframe = soup.find('iframe', id='pdf')
                if iframe and 'src' in iframe.attrs:
                    pdf_url = iframe['src']
                    if pdf_url.startswith('//'):
                        pdf_url = 'https:' + pdf_url
                    elif not pdf_url.startswith('http'):
                        pdf_url = urljoin(domain, pdf_url)
                        
                    print(f"Found PDF URL: {pdf_url}")
                    
                    # Download the PDF
                    pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=30)
                    
                    if pdf_response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(pdf_response.content)
                            
                        # Verify the file was downloaded and is a valid PDF
                        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                            # Create a better filename
                            if article.title:
                                safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '' for c in article.title])
                                safe_title = safe_title[:100]  # Limit length
                                new_filename = f"{pmid}_{safe_title}.pdf"
                                os.rename(pdf_path, os.path.join(RESULTS_FOLDER, new_filename))
                                
                                # Record the download in the TSV file
                                with open(DOWNLOADED_PUBMEDS_FILEPATH, 'a') as f:
                                    f.write(f"{pmid}\t\t{article.title}\n")
                                    
                                print(f"Successfully downloaded from Sci-Hub for PMID {pmid}")
                                return True
            except Exception as e:
                print(f"Error with Sci-Hub domain {domain} for PMID {pmid}: {str(e)}")
                continue
                
        return False
    except Exception as e:
        print(f"Error using Sci-Hub for PMID {pmid}: {str(e)}")
        return False

def get_last_id_from_metadata():
    """
    Get the last Sequential ID from the metadata file.
    """
    if os.path.exists(METADATA_FILEPATH):
        with open(METADATA_FILEPATH, 'r') as f:
            metadata = json.load(f)
        # Get the maximum id in the current metadata
        max_id = max([entry.get('id', 0) for entry in metadata], default=0)
    else:
        max_id = 0  # If the metadata file doesn't exist, start from 0
    return max_id


def append_to_metadata_file(task):
    """
    Update the metadata file with the new PubMed article entry.
    """
    try:
        # Load existing metadata
        if os.path.exists(METADATA_FILEPATH):
            with open(METADATA_FILEPATH, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []

        # Append the new task to the metadata
        metadata.append(task)

        # Write back to the file
        with open(METADATA_FILEPATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata updated for article ID: {task['id']}")
    except Exception as e:
        print(f"Error updating metadata file: {e}")


def update_article_metadata(pmid, task_id):
    """
    Generate and update metadata for a downloaded article.
    """
    pdf_path = os.path.join(RESULTS_FOLDER, f"{pmid}.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"PDF file does not exist for PMID: {pmid}. Skipping metadata update.")
        return False

    print(f"Processing metadata for PMID: {pmid}")
    try:
        # Get article metadata from PubMed
        article = FETCH.article_by_pmid(pmid)
        if not article:
            print(f"Failed to fetch metadata for PMID: {pmid}.")
            return False
            
        # Calculate MD5 hash
        md5_hash = calculate_md5(pdf_path)
        
        # Get PDF page count
        page_count = get_pdf_page_length(pdf_path)
        
        # Get author h-index (for first author)
        author_h_index = 0
        if hasattr(article, 'authors') and article.authors:
            first_author = article.authors[0] if isinstance(article.authors, list) else article.authors
            if isinstance(first_author, str):
                author_name = first_author
            elif isinstance(first_author, dict):
                author_name = first_author.get('name', '')
            else:
                author_name = str(first_author)
            author_h_index = get_author_h_index(author_name)
        
        # Get journal impact factor
        journal_impact_factor = "Not available"
        if hasattr(article, 'journal'):
            journal_impact_factor = get_journal_impact_factor(article.journal)
        
        # Create embedded PDF field
        pdf_embed = f"<embed src='/data/local-files/?d=pdfs/{pmid}.pdf' width='100%' height='600px'/>"
        
        # Create metadata entry
        task = {
            "id": task_id,
            "filename": f"{pmid}.pdf",
            "title": article.title if hasattr(article, 'title') else "Title not available",
            "document_id": pmid,
            "url": article.url if hasattr(article, 'url') else f"{PUBMED_BASE_URL}{pmid}",
            "publication_date": str(article.history.get("accepted")) if hasattr(article, 'history') else "",
            "abstract": article.abstract if hasattr(article, 'abstract') else "Abstract not available",
            "authors": article.authors if hasattr(article, 'authors') else [],
            "tags": article.keywords if hasattr(article, 'keywords') else [],
            "pages": page_count,
            "md5_hash": md5_hash,
            "author_h_index": author_h_index,
            "journal_impact_factor": journal_impact_factor,
            "pdf": pdf_embed,
            "curator_id": "AlNu-Health",
            "last_updated_date": time.strftime("%Y-%m-%d"),
            "full_path": os.path.abspath(pdf_path)
        }
        
        # Append to metadata file
        append_to_metadata_file(task)
        return True
    except Exception as e:
        print(f"Failed to update metadata for PMID: {pmid}. Error: {str(e)}")
        return False


def download_pubmed_articles(pmids, progress_bar=None):
    """
    Download the specified PubMed articles using multiple methods.
    """
    downloaded_count = 0
    task_id = get_last_id_from_metadata() + 1  # Initialize task ID based on last existing ID
    
    for i, pmid in enumerate(pmids):
        if progress_bar:
            progress_bar.progress((i + 1) / len(pmids), text=f"Downloading article {i+1}/{len(pmids)}")
        
        if is_article_already_downloaded(pmid):
            print(f"PMID {pmid} already exists. Skipping download.")
            downloaded_count += 1
            continue  # Skip download if already exists

        print(f"Downloading article for PMID: {pmid}")
        success = False
        
        # Method 1: Try to get PMC ID and download directly from PMC
        pmc_id = get_pmc_id_from_pmid(pmid)
        if pmc_id:
            print(f"Found PMC ID {pmc_id} for PMID {pmid}")
            success = download_from_pmc_direct(pmid, pmc_id)
        
        # Method 2: If PMC direct download failed, try pubmed2pdf
        if not success:
            try:
                temp_error_file = os.path.join(RESULTS_FOLDER, f"pubmed2pdf_errors_{pmid}.txt")
                result = subprocess.run(
                    [sys.executable, "-m", "pubmed2pdf", "pdf", 
                     f"--out={RESULTS_FOLDER}", 
                     f"--errors={temp_error_file}", 
                     f"--pmids={pmid}"],
                    capture_output=True,
                    text=True,
                    timeout=60  # Set a timeout to avoid hanging
                )
                
                # Check if the download was successful
                if is_article_already_downloaded(pmid):
                    success = True
                    # Rename the file to include the PMID for easier identification
                    article = FETCH.article_by_pmid(pmid)
                    if article and hasattr(article, 'title'):
                        # Create a safe filename from the title
                        safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '' for c in article.title])
                        safe_title = safe_title[:100]  # Limit length
                        new_filename = f"{pmid}_{safe_title}.pdf"
                        os.rename(
                            os.path.join(RESULTS_FOLDER, f"{pmid}.pdf"),
                            os.path.join(RESULTS_FOLDER, new_filename)
                        )
                        
                        # Record the download in the TSV file
                        with open(DOWNLOADED_PUBMEDS_FILEPATH, 'a') as f:
                            f.write(f"{pmid}\t{pmc_id or ''}\t{article.title}\n")
                
                # Check and process error file
                if os.path.exists(temp_error_file):
                    with open(temp_error_file, 'r') as f:
                        error_pmids = f.read().splitlines()
                    for error_pmid in error_pmids:
                        append_to_error_file(error_pmid)
                    # Remove the temp error file
                    os.remove(temp_error_file)
            except Exception as e:
                print(f"Error using pubmed2pdf for PMID {pmid}: {str(e)}")
        
        # Method 3: Try Sci-Hub as a last resort
        if not success:
            success = download_using_scihub(pmid)
        
        if success:
            # Update metadata for the downloaded article
            metadata_success = update_article_metadata(pmid, task_id)
            if metadata_success:
                task_id += 1  # Increment task ID only if metadata was successfully updated
                
            downloaded_count += 1
            print(f"Successfully downloaded PMID {pmid}")
        else:
            print(f"Failed to download PMID: {pmid}")
            append_to_error_file(pmid)
    
    return downloaded_count

def download_articles_by_keywords(keywords, max_articles_per_keyword=5, filter_open_access=True, year_range=None):
    """
    For the given keywords, identify and download PubMed articles.
    Optionally filter for open access articles and specific year range.
    """
    initial_setup()
    total_downloaded = 0
    
    for keyword in keywords:
        start_time = time.time()
        
        # Modify search term with year range if provided
        search_term = keyword
        if year_range:
            search_term = f"{keyword} AND ({year_range[0]}[PDAT] : {year_range[1]}[PDAT])"
        
        # Search PubMed for articles matching the keyword
        pmids = get_pubmed_search_query(search_term, max_articles_per_keyword, filter_open_access)
        
        if not pmids:
            print(f"No articles found for keyword '{keyword}'")
            continue
            
        print(f"Found {len(pmids)} articles for keyword '{keyword}'.")
        
        # Download the articles
        downloaded = download_pubmed_articles(pmids)
        total_downloaded += downloaded
        
        print(f"Downloaded {downloaded} articles for keyword '{keyword}'")
        print(f"Time to Process: {time.time() - start_time:.2f} secs")
        
        # Update the download status in the overall status file
        try:
            if os.path.exists(OVERALL_DOWNLOAD_STATUS_FILEPATH):
                with open(OVERALL_DOWNLOAD_STATUS_FILEPATH, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
                
            # Update status for this keyword
            status[keyword] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "found": len(pmids),
                "downloaded": downloaded,
                "filter_open_access": filter_open_access,
                "year_range": year_range
            }
            
            # Write back to the file
        except Exception as e:
            print(f"Error updating download status: {e}")
    
    return total_downloaded

def download_articles_by_keywords(keywords, max_articles_per_keyword=5, filter_open_access=True, year_range=None):
    """
For the given keywords, identify and download PubMed articles.
Optionally filter for open access articles and specific year range.
"""
    initial_setup()
    total_downloaded = 0
    
    for keyword in keywords:
        start_time = time.time()
        
        # Modify search term with year range if provided
        search_term = keyword
        if year_range:
            search_term = f"{keyword} AND ({year_range[0]}[PDAT] : {year_range[1]}[PDAT])"
        
        # Search PubMed for articles matching the keyword
        pmids = get_pubmed_search_query(search_term, max_articles_per_keyword, filter_open_access)
        
        if not pmids:
            print(f"No articles found for keyword '{keyword}'")
            continue
                
        print(f"Found {len(pmids)} articles for keyword '{keyword}'.")
        
        # Download the articles
        downloaded = download_pubmed_articles(pmids)
        total_downloaded += downloaded
            
        print(f"Downloaded {downloaded} articles for keyword '{keyword}'")
        print(f"Time to Process: {time.time() - start_time:.2f} secs")
        
        # Update the download status in the overall status file
        try:
            if os.path.exists(OVERALL_DOWNLOAD_STATUS_FILEPATH):
                with open(OVERALL_DOWNLOAD_STATUS_FILEPATH, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
                    
            # Update status for this keyword
            status[keyword] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "found": len(pmids),
                "downloaded": downloaded,
                "filter_open_access": filter_open_access,
                "year_range": year_range
            }
                
            # Write back to the file
            with open(OVERALL_DOWNLOAD_STATUS_FILEPATH, 'w') as f:
                json.dump(status, f, indent=4)
        except Exception as e:
            print(f"Error updating download status: {e}")
    
    return total_downloaded

# Import necessary functions for vector database integration
from document_processor import process_documents
from vector_db import create_vector_db

# Streamlit interface for the PubMed downloader
def pubmed_downloader_ui():
    """Streamlit interface for the PubMed downloader"""
    st.title("PubMed Article Downloader")
    
    # Initialize session state variables
    if 'download_status' not in st.session_state:
        st.session_state.download_status = None
    if 'downloaded_count' not in st.session_state:
        st.session_state.downloaded_count = 0
    
    # Setup tabs
    tab1, tab2, tab3 = st.tabs(["Search & Download", "Manage Downloads", "Metadata Explorer"])
    
    with tab1:
        st.header("Search & Download Articles")
        
        # Input for keywords
        keywords_input = st.text_area("Enter keywords (one per line)", height=100)
        
        # Number of articles per keyword
        max_articles = st.slider("Maximum articles per keyword", 1, 20, 5)
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_open_access = st.checkbox("Filter for Open Access articles only", value=True)
        with col2:
            include_year_range = st.checkbox("Filter by publication year", value=True)
        
        # Year range selector (only shown if year filter is enabled)
        year_range = None
        if include_year_range:
            year_col1, year_col2 = st.columns(2)
            with year_col1:
                start_year = st.number_input("Start Year", min_value=1900, max_value=2025, value=2010)
            with year_col2:
                end_year = st.number_input("End Year", min_value=1900, max_value=2025, value=2023)
            year_range = (start_year, end_year)
        
        # Download button
        if st.button("Download Articles"):
            if keywords_input.strip():
                keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
                
                if keywords:
                    with st.spinner("Downloading articles..."):
                        progress_bar = st.progress(0)
                        
                        # Download articles with year range if specified
                        st.session_state.downloaded_count = download_articles_by_keywords(
                            keywords, 
                            max_articles, 
                            filter_open_access,
                            year_range
                        )
                        
                        # Update status
                        st.session_state.download_status = "completed"
                        progress_bar.progress(1.0)
                else:
                    st.error("Please enter at least one keyword.")
            else:
                st.error("Please enter at least one keyword.")
    
    with tab2:
        st.header("Manage Downloaded Articles")
        
        # Display download status
        if st.session_state.download_status == "completed":
            st.success(f"Downloaded {st.session_state.downloaded_count} articles.")
            
        # Maintenance & Vector DB Integration section - MOVED TO TOP
        st.subheader("Maintenance & Vector DB Integration")
        
        # Get papers not in the vector database
        pdf_files = [f for f in os.listdir(RESULTS_FOLDER) if f.lower().endswith('.pdf')]
        total_papers = len(pdf_files)
        
        # Get list of documents already in the vector database
        vectordb_docs = set()
        if 'chroma_instance' in st.session_state and st.session_state.chroma_instance is not None:
            try:
                # Get all document IDs from the vector database
                collection = st.session_state.chroma_instance._collection
                if collection:
                    # Get all metadatas
                    result = collection.get(include=["metadatas"])
                    if result and "metadatas" in result and result["metadatas"]:
                        metadatas = result["metadatas"]
                        # Extract unique document IDs
                        for metadata in metadatas:
                            if metadata and 'document_id' in metadata:
                                vectordb_docs.add(metadata['document_id'])
            except Exception as e:
                st.error(f"Error getting documents from vector database: {e}")
        
        # Find PDFs not in the vector database
        new_pdfs = [pdf for pdf in pdf_files if pdf not in vectordb_docs]
        db_papers_count = total_papers - len(new_pdfs)
        
        # Display counts
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total downloaded papers:** {total_papers}")
            st.info(f"**Papers in vector database:** {db_papers_count}")
            
            if os.path.exists(PUBMED_ERRORS_FILEPATH):
                if st.button("Clear Error Log"):
                    with open(PUBMED_ERRORS_FILEPATH, 'w') as f:
                        f.write('')
                    st.success("Error log cleared.")
        
        with col2:
            if len(new_pdfs) > 0:
                st.warning(f"**{len(new_pdfs)} papers not in vector database**")
                
                if st.button("Add Missing Papers to Vector DB"):
                    with st.spinner(f"Adding {len(new_pdfs)} papers to vector database..."):
                        # Only process the missing papers instead of all documents
                        new_docs = []
                        results_dir = RESULTS_FOLDER
                        
                        # Create a progress bar for processing
                        progress_bar = st.progress(0)
                        
                        # Process each missing PDF file directly
                        for i, pdf_file in enumerate(new_pdfs):
                            # Update progress
                            progress_percent = i / len(new_pdfs)
                            progress_bar.progress(progress_percent)
                            
                            pdf_path = os.path.join(results_dir, pdf_file)
                            try:
                                # Extract text from the PDF
                                from document_processor import extract_text_from_pdf
                                text = extract_text_from_pdf(pdf_path)
                                
                                # Create document info
                                doc_info = {"name": pdf_file, "content": text, "path": pdf_path}
                                new_docs.append(doc_info)
                                
                            except Exception as e:
                                st.error(f"Error processing {pdf_file}: {str(e)}")
                        
                        # Complete the progress bar
                        progress_bar.progress(1.0)
                        
                        if new_docs:
                            # Add documents to vector database
                            vectorstore = create_vector_db(new_docs, update_existing=True)
                            if vectorstore:
                                st.success(f"Successfully added {len(new_docs)} papers to the vector database!")
                                # Use st.rerun() instead of the deprecated st.experimental_rerun()
                                st.rerun()
                            else:
                                st.error("Failed to add papers to vector database.")
            else:
                st.success("All papers are in the vector database")
            
            if st.button("Process Metadata for All Articles"):
                with st.spinner("Processing metadata..."):
                    # Get all PDF files
                    pdf_files = [f for f in os.listdir(RESULTS_FOLDER) if f.lower().endswith('.pdf')]
                    task_id = get_last_id_from_metadata() + 1
                    processed_count = 0
                    
                    for pdf_file in pdf_files:
                        # Extract PMID from filename
                        pmid = pdf_file.split('_')[0] if '_' in pdf_file else pdf_file.replace('.pdf', '')
                        if pmid.isdigit():
                            success = update_article_metadata(pmid, task_id)
                            if success:
                                task_id += 1
                                processed_count += 1
                    
                    st.success(f"Processed metadata for {processed_count} articles.")
        
        # List downloaded articles section - MOVED BELOW MAINTENANCE
        st.subheader(f"Downloaded Articles ({total_papers})")
        
        # Add search filter
        search_term = st.text_input("Filter articles by name", "")
        
        # Filter files based on search term
        display_files = pdf_files
        if search_term:
            display_files = [f for f in pdf_files if search_term.lower() in f.lower()]
        
        # Sort files by date (newest first)
        display_files = sorted(display_files, key=lambda f: os.path.getmtime(os.path.join(RESULTS_FOLDER, f)), reverse=True)
        
        if display_files:
            for pdf_file in display_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(pdf_file)
                with col2:
                    st.download_button(
                        label="Download",
                        data=open(os.path.join(RESULTS_FOLDER, pdf_file), "rb").read(),
                        file_name=pdf_file,
                        mime="application/pdf"
                    )
                with col3:
                    # Extract PMID from filename
                    pmid = pdf_file.split('_')[0] if '_' in pdf_file else pdf_file.replace('.pdf', '')
                    st.markdown(f"[View on PubMed]({PUBMED_BASE_URL}{pmid})")
        else:
            st.info("No articles found matching your filter.")
                


    
    with tab3:
        st.header("Metadata Explorer")
        
        if os.path.exists(METADATA_FILEPATH):
            try:
                with open(METADATA_FILEPATH, 'r') as f:
                    metadata = json.load(f)
                
                if metadata:
                    st.subheader(f"Article Metadata ({len(metadata)} entries)")
                    
                    # Add search filter for metadata
                    search_term = st.text_input("Search metadata by title or author", "", key="metadata_search")
                    
                    # Filter metadata based on search term
                    if search_term:
                        filtered_metadata = []
                        for entry in metadata:
                            title = entry.get('title', '').lower()
                            authors = str(entry.get('authors', '')).lower()
                            if search_term.lower() in title or search_term.lower() in authors:
                                filtered_metadata.append(entry)
                        display_metadata = filtered_metadata
                    else:
                        display_metadata = metadata
                    
                    # Sort by ID (newest first)
                    display_metadata = sorted(display_metadata, key=lambda x: x.get('id', 0), reverse=True)
                    
                    # Display metadata entries
                    for entry in display_metadata:
                        with st.expander(f"{entry.get('id', 'Unknown ID')}: {entry.get('title', 'Unknown Title')}"):
                            st.markdown(f"**Document ID:** {entry.get('document_id', 'N/A')}")
                            st.markdown(f"**Publication Date:** {entry.get('publication_date', 'N/A')}")
                            st.markdown(f"**Authors:** {', '.join(entry.get('authors', [])) if isinstance(entry.get('authors', []), list) else entry.get('authors', 'N/A')}")
                            st.markdown(f"**Journal Impact Factor:** {entry.get('journal_impact_factor', 'N/A')}")
                            st.markdown(f"**First Author H-Index:** {entry.get('author_h_index', 'N/A')}")
                            st.markdown(f"**Pages:** {entry.get('pages', 'N/A')}")
                            st.markdown(f"**Tags:** {', '.join(entry.get('tags', [])) if isinstance(entry.get('tags', []), list) else entry.get('tags', 'N/A')}")
                            st.markdown(f"**Abstract:**")
                            st.markdown(entry.get('abstract', 'No abstract available'))
                            
                            # Add link to PubMed
                            st.markdown(f"[View on PubMed]({entry.get('url', '#')})")
                            
                            # Add PDF viewer if file exists
                            pdf_path = entry.get('full_path')
                            if pdf_path and os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_bytes,
                                    file_name=entry.get('filename', 'article.pdf'),
                                    mime="application/pdf"
                                )
                else:
                    st.info("No metadata entries found. Download some articles first.")
            except Exception as e:
                st.error(f"Error loading metadata: {str(e)}")
        else:
            st.info("No metadata file found. Download some articles first.")

if __name__ == "__main__":
    download_articles_by_keywords(["Invasive Breast Cancer", "Inflammatory Breast Cancer"], 5)
