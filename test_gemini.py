import streamlit as st
import google.generativeai as genai
from gemini_embeddings import GeminiEmbeddings
from gemini_llm import GeminiLLM
from config import GEMINI_API_KEY, GEMINI_CONFIG

def main():
    st.title("Gemini API Test")
    
    # Display API key status
    if GEMINI_API_KEY:
        st.success("Gemini API key is configured")
    else:
        st.error("Gemini API key is not configured. Please add it to your secrets.toml file.")
        return
    
    # Test direct Gemini API
    st.header("1. Direct Gemini API Test")
    
    try:
        # Initialize the Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Generate content
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Explain how AI works in a few words")
        
        # Display the response
        st.write("Response from direct Gemini API:")
        st.write(response.text)
        st.success("Direct Gemini API test successful!")
    except Exception as e:
        st.error(f"Error testing direct Gemini API: {str(e)}")
    
    # Test Gemini embeddings
    st.header("2. Gemini Embeddings Test")
    
    try:
        # Initialize embeddings
        embedding_model = GeminiEmbeddings(api_key=GEMINI_API_KEY)
        
        # Get embeddings for a test query
        test_query = "How does machine learning work?"
        embeddings = embedding_model.embed_query(test_query)
        
        # Display embedding info
        st.write(f"Generated embeddings for: '{test_query}'")
        st.write(f"Embedding dimension: {len(embeddings)}")
        st.write(f"First 5 values: {embeddings[:5]}")
        st.success("Gemini embeddings test successful!")
    except Exception as e:
        st.error(f"Error testing Gemini embeddings: {str(e)}")
    
    # Test Gemini LLM with simple prompt
    st.header("3. Gemini LLM - Simple Prompt Test")
    
    try:
        # Initialize LLM
        llm = GeminiLLM()
        
        # Generate a response
        test_prompt = "Explain the benefits of a plant-based diet in 3 sentences."
        response = llm(test_prompt)
        
        # Display the response
        st.write(f"Prompt: '{test_prompt}'")
        st.write("Response:")
        st.write(response)
        st.success("Gemini LLM simple prompt test successful!")
    except Exception as e:
        st.error(f"Error testing Gemini LLM with simple prompt: {str(e)}")
    
    # Test Gemini LLM with chat messages
    st.header("4. Gemini LLM - Chat Messages Test")
    
    try:
        # Initialize LLM
        llm = GeminiLLM()
        
        # Create messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specialized in nutrition."},
            {"role": "user", "content": "What are good sources of vitamin B12?"}
        ]
        
        # Generate a response
        response = llm(messages)
        
        # Display the response
        st.write("Messages:")
        for msg in messages:
            st.write(f"- {msg['role']}: {msg['content']}")
        st.write("Response:")
        st.write(response)
        st.success("Gemini LLM chat messages test successful!")
    except Exception as e:
        st.error(f"Error testing Gemini LLM with chat messages: {str(e)}")

if __name__ == "__main__":
    main()
