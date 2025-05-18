import os
import json
import time
import streamlit as st
from Bio import Entrez
from config import EMAIL_ID

# Set up Entrez email
Entrez.email = EMAIL_ID

def check_keyword_availability(keyword, max_results=10000, filter_open_access=True):
    """
    Check the availability of papers for a given keyword in PubMed.
    
    Args:
        keyword (str): The keyword to search for
        max_results (int): Maximum number of results to return
        filter_open_access (bool): Whether to filter for open access articles only
        
    Returns:
        dict: A dictionary containing the search results
    """
    try:
        # Add open access filter if requested
        search_term = keyword
        if filter_open_access:
            search_term = f"{search_term} AND (open access[Filter] OR free full text[Filter])"
            
        # Search PubMed for the given term
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record.get("IdList", [])
        count = int(record.get("Count", 0))
        
        return {
            "keyword": keyword,
            "total_count": count,
            "retrieved_count": len(pmids),
            "pmids": pmids,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filter_open_access": filter_open_access
        }
    except Exception as e:
        st.error(f"Error searching PubMed for '{keyword}': {str(e)}")
        return {
            "keyword": keyword,
            "total_count": 0,
            "retrieved_count": 0,
            "pmids": [],
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filter_open_access": filter_open_access
        }

def batch_check_keywords(keywords, max_results=10000, filter_open_access=True):
    """
    Check the availability of papers for multiple keywords in PubMed.
    
    Args:
        keywords (list): List of keywords to search for
        max_results (int): Maximum number of results to return per keyword
        filter_open_access (bool): Whether to filter for open access articles only
        
    Returns:
        dict: A dictionary mapping keywords to their availability data
    """
    results = {}
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, keyword in enumerate(keywords):
        # Update progress
        progress = (i + 1) / len(keywords)
        progress_bar.progress(progress)
        status_text.text(f"Checking availability for: {keyword}")
        
        # Check availability
        result = check_keyword_availability(keyword, max_results, filter_open_access)
        results[keyword] = result
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def keyword_availability_ui():
    """
    Streamlit interface for checking keyword availability
    """
    st.title("Research Paper Availability")
    
    # Define the keyword categories and their keywords
    keyword_categories = {
        "Core Scientific Keywords": [
            "Glucagon-Like Peptide-1 OR GLP1RAs OR GLP-1",
            "semaglutide OR liraglutide OR tirzepatide OR dulaglutide OR retatrutide OR cagrisema",
            "GLP-1 AND muscle mass",
            "GLP-1 AND side effects OR GLP-1 AND adverse events",
            "GLP-1 AND physical activity",
            "GLP-1 AND behavioral intervention",
            "weight loss medication AND personalized nutrition",
            "GLP-1 AND gastrointestinal symptoms",
            "GLP-1 AND Obesity",
            "GLP-1 and Type 2 Diabetes",
            "GLP-1 and T2DM",
            "GLP-1 and Cardiovascular Disease",
            "GLP-1/GIP dual agonist and obstructive sleep apnea",
            "GLP-1 and VAT",
            "GLP-1 and Visceral Adipose Tissue",
            "GLP-1 and waist circumference",
            "GLP-1 and hair loss"
        ],
        "Nutrition & Diet Keywords": [
            "GLP-1 AND dietary intervention",
            "macronutrient composition AND GLP-1",
            "GLP-1 AND protein supplementation",
            "GLP-1 AND fiber",
            "GLP-1 AND water intake",
            "nutrition AND weight loss",
            "meal timing AND GLP-1"
        ],
        "Muscle Retention + Fitness Keywords": [
            "sarcopenia AND GLP-1",
            "GLP-1 AND lean muscle mass",
            "GLP-1 AND Muscle Loss",
            "GLP-1 AND strength training",
            "GLP-1 AND resistance exercise",
            "GLP-1 AND cardiovascular exercise",
            "muscle loss AND obesity",
            "exercise intervention AND GLP-1",
            "Lifestyle intervention AND GLP-1"
        ]
    }
    
    # Filter options
    st.subheader("Search Options")
    filter_open_access = st.checkbox("Filter for Open Access articles only", value=True)
    
    # Button to check availability
    if st.button("Check Paper Availability"):
        # Flatten the keywords list
        all_keywords = []
        for category, keywords in keyword_categories.items():
            all_keywords.extend(keywords)
        
        # Check availability for all keywords
        with st.spinner("Checking paper availability..."):
            results = batch_check_keywords(all_keywords, 10000, filter_open_access)
            
            # Store results in session state
            st.session_state.keyword_availability_results = results
    
    # Display results if available
    if 'keyword_availability_results' in st.session_state:
        results = st.session_state.keyword_availability_results
        
        # Display results by category
        for category, keywords in keyword_categories.items():
            st.subheader(category)
            
            # Create a table for this category
            table_data = []
            for keyword in keywords:
                if keyword in results:
                    result = results[keyword]
                    table_data.append({
                        "Keyword": keyword,
                        "Total Papers": result["total_count"],
                        "Open Access" if filter_open_access else "Available": result["retrieved_count"]
                    })
            
            # Display the table
            if table_data:
                st.table(table_data)
            else:
                st.info(f"No results available for {category}")
            
            # Add a divider
            st.markdown("---")

if __name__ == "__main__":
    keyword_availability_ui()
