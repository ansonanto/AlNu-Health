import io
import base64
import logging
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from config import GEMINI_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API - try multiple sources for the API key
def configure_gemini_api():
    # Try getting the API key from different sources
    api_key = None
    
    # 1. Try from environment variable directly
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # 2. Try from config module
    if not api_key:
        api_key = GEMINI_API_KEY
        logger.info(f"Using API key from config: {'Found' if api_key else 'Not found'}")
    
    # 3. Try from Streamlit secrets directly
    if not api_key:
        try:
            api_key = st.secrets["api_keys"]["gemini"]
            logger.info("Using API key from Streamlit secrets")
        except Exception as e:
            logger.error(f"Error getting API key from Streamlit secrets: {str(e)}")
    
    # Configure the API if we found a key
    if api_key:
        logger.info("Configuring Gemini API with key")
        genai.configure(api_key=api_key)
        return True
    else:
        logger.error("No Gemini API key found in any source")
        return False

# Configure the API when module is loaded
configure_gemini_api()

def is_food_image(image_data):
    """
    Check if the uploaded image contains food
    
    Args:
        image_data: Image data in bytes
        
    Returns:
        bool: True if food is detected, False otherwise
    """
    try:
        # Directly use the API key from Streamlit secrets
        api_key = st.secrets["api_keys"]["gemini"]
        logger.info(f"Using API key from Streamlit secrets: {api_key[:5]}...")
        genai.configure(api_key=api_key)
            
        # Configure the model - use Gemini 1.5 Flash instead of deprecated Pro Vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Using Gemini 1.5 Flash model for food detection")
        
        # Convert image data to format expected by Gemini
        img = Image.open(io.BytesIO(image_data))
        
        # Create prompt for food detection
        prompt = "Is this image of food? Please respond with only 'yes' or 'no'."
        
        # Generate response
        logger.info("Sending request to Gemini API for food detection")
        response = model.generate_content([prompt, img])
        result = response.text.strip().lower()
        logger.info(f"Gemini API response for food detection: {result}")
        
        # Check if the response contains 'yes'
        is_food = 'yes' in result
        logger.info(f"Food detected: {is_food}")
        return is_food
    except Exception as e:
        logger.error(f"Error detecting food in image: {str(e)}")
        return False

def extract_macros(image_data):
    """
    Extract macronutrient information from a food image
    
    Args:
        image_data: Image data in bytes
        
    Returns:
        dict: Dictionary containing macronutrient information
    """
    try:
        # Directly use the API key from Streamlit secrets
        api_key = st.secrets["api_keys"]["gemini"]
        logger.info(f"Using API key from Streamlit secrets: {api_key[:5]}...")
        genai.configure(api_key=api_key)
        
        # Configure the model - use Gemini 1.5 Flash instead of deprecated Pro Vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Using Gemini 1.5 Flash model for macro extraction")
        
        # Convert image data to format expected by Gemini
        img = Image.open(io.BytesIO(image_data))
        
        # Create prompt for macro extraction
        prompt = """
        Analyze this food image and provide the following nutritional information:
        1. Food name
        2. Calories (kcal)
        3. Protein (g)
        4. Carbohydrates (g)
        5. Fat (g)
        6. Fiber (g)
        
        Format your response as a JSON object with the following keys:
        {
            "food_name": "Name of the food",
            "calories": number,
            "protein": number,
            "carbs": number,
            "fat": number,
            "fiber": number
        }
        
        Provide only the JSON object with no additional text.
        """
        
        # Generate response
        logger.info("Sending request to Gemini API for macro extraction")
        response = model.generate_content([prompt, img])
        result = response.text.strip()
        logger.info(f"Received response from Gemini API: {result[:50]}...")
        
        # Extract JSON from response (in case there's additional text)
        import json
        import re
        
        # Find JSON pattern in the response
        json_match = re.search(r'({[\s\S]*})', result)
        if json_match:
            json_str = json_match.group(1)
            # Parse JSON
            try:
                macros = json.loads(json_str)
                logger.info(f"Successfully extracted macros: {macros.get('food_name', 'Unknown')}")
                return macros
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from response")
                return None
        else:
            logger.error("No JSON found in response")
            return None
    except Exception as e:
        logger.error(f"Error extracting macros from image: {str(e)}")
        return None

def format_macro_display(macros):
    """
    Format macronutrient information for display using Streamlit native components
    
    Args:
        macros: Dictionary containing macronutrient information
        
    Returns:
        None: This function directly renders to the Streamlit UI
    """
    import streamlit as st
    
    if not macros:
        st.error("Unable to extract macronutrient information.")
        return
    
    # Calculate percentages for progress bars (based on typical daily values)
    cal_percent = min(100, int(macros.get('calories', 0) / 2000 * 100)) / 100
    protein_percent = min(100, int(macros.get('protein', 0) / 50 * 100)) / 100
    carbs_percent = min(100, int(macros.get('carbs', 0) / 300 * 100)) / 100
    fat_percent = min(100, int(macros.get('fat', 0) / 65 * 100)) / 100
    fiber_percent = min(100, int(macros.get('fiber', 0) / 25 * 100)) / 100
    
    # Use Streamlit's native components for better rendering
    st.markdown(f"### {macros.get('food_name', 'Unknown Food')}")
    
    # Use columns for the layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Macronutrients")
        
        # Calories
        st.markdown(f"**Calories:** {macros.get('calories', 'N/A')} kcal")
        st.progress(cal_percent, text="")
        
        # Protein
        st.markdown(f"**Protein:** {macros.get('protein', 'N/A')} g")
        st.progress(protein_percent, text="")
        
        # Carbohydrates
        st.markdown(f"**Carbohydrates:** {macros.get('carbs', 'N/A')} g")
        st.progress(carbs_percent, text="")
        
        # Fat
        st.markdown(f"**Fat:** {macros.get('fat', 'N/A')} g")
        st.progress(fat_percent, text="")
        
        # Fiber
        st.markdown(f"**Fiber:** {macros.get('fiber', 'N/A')} g")
        st.progress(fiber_percent, text="")
    
    with col2:
        # Create a pie chart for macronutrient distribution
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get values for the pie chart
        protein = float(macros.get('protein', 0)) * 4  # 4 calories per gram
        carbs = float(macros.get('carbs', 0)) * 4     # 4 calories per gram
        fat = float(macros.get('fat', 0)) * 9         # 9 calories per gram
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(4, 4))
        sizes = [protein, carbs, fat]
        labels = ['Protein', 'Carbs', 'Fat']
        colors = ['#42A5F5', '#EC407A', '#FFEE58']
        
        # Only show the pie chart if we have valid data
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Calorie Distribution')
            st.pyplot(fig)
            
            # Show detailed calorie breakdown
            st.markdown("### Calorie Breakdown")
            st.markdown(f"""                
            - Protein: {macros.get('protein', 0)}g × 4 kcal = {protein:.0f} kcal
            - Carbs: {macros.get('carbs', 0)}g × 4 kcal = {carbs:.0f} kcal
                - of which Fiber: {macros.get('fiber', 0)}g (included in total carbs)
            - Fat: {macros.get('fat', 0)}g × 9 kcal = {fat:.0f} kcal
            - **Total: {protein + carbs + fat:.0f} kcal**
            """)
            
            st.info("Note: Fiber is a type of carbohydrate and is already included in the total carbs. While it contributes to the carb count, it has different digestive and nutritional properties.")
        else:
            st.warning("Not enough data for calorie distribution chart")
    
    # Add a note about daily values
    st.caption("Percentages based on a 2000 calorie diet")
    
    # Return None since we're directly rendering to the UI
    return None

def process_food_image(uploaded_file):
    """
    Process an uploaded food image
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (success, message, macros)
    """
    try:
        # Read image data
        image_data = uploaded_file.getvalue()
        
        # Check if the image contains food
        if not is_food_image(image_data):
            return False, "The uploaded image does not appear to contain food.", None
        
        # Extract macronutrient information
        macros = extract_macros(image_data)
        
        if macros:
            return True, "Successfully analyzed food image.", macros
        else:
            return False, "Failed to extract macronutrient information.", None
    except Exception as e:
        logger.error(f"Error processing food image: {str(e)}")
        return False, f"Error processing image: {str(e)}", None
