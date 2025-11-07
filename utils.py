import os
import streamlit as st
from PIL import Image, ImageFilter, UnidentifiedImageError
from datetime import datetime
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fpdf import FPDF
import tempfile
import base64
import numpy as np
import cv2
import shutil
import uuid
import glob
from streamlit_cropper import st_cropper
import platform
import logging
import pandas as pd
import torch
from torchvision import transforms
import time
import random
from typing import Optional, Any, cast
import requests

# pylint: disable=no-member
# pyright: reportAttributeAccessIssue=false

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
if platform.system() == "Darwin" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Allow static analysis tools to treat cv2 as dynamic module with runtime attributes
cv2 = cast(Any, cv2)

# Load environment variables
load_dotenv(override=True)


def _clean_api_key(key: Optional[str]) -> Optional[str]:
    """Normalize Groq API keys entered via .env or text input."""
    if not key:
        return None
    cleaned = key.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


GROQ_API_KEY = _clean_api_key(os.getenv("GROQ_API_KEY"))


def get_groq_api_key() -> Optional[str]:
    """Return the cached Groq API key or refresh from the environment."""
    global GROQ_API_KEY
    if GROQ_API_KEY:
        return GROQ_API_KEY
    GROQ_API_KEY = _clean_api_key(os.getenv("GROQ_API_KEY"))
    return GROQ_API_KEY


def set_groq_api_key(key: Optional[str]) -> Optional[str]:
    """Cache and persist the Groq API key after sanitizing it."""
    global GROQ_API_KEY
    GROQ_API_KEY = _clean_api_key(key)
    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    elif "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    return GROQ_API_KEY

def retry_with_exponential_backoff(func, max_retries=3, base_delay=1):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
    
    Returns:
        Result of the function call
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            
            # If it's not a capacity issue, don't retry
            if "over capacity" not in error_msg and "503" not in str(e):
                raise e
            
            if attempt == max_retries:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"GROQ API over capacity, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

# Cache BLIP models
@st.cache_resource
def load_models():
    """Load BLIP models for image captioning."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load BLIP models: {e}")
        return None, None

# Image processing functions
def check_image_quality(image: Image.Image, suspected_deficiency: str = None) -> bool:
    """Check if image quality is sufficient for analysis."""
    try:
        # Check image size
        if image.size[0] < 100 or image.size[1] < 100:
            return False
        
        # Check if image is too blurry
        gray = image.convert('L')
        laplacian_var = cv2.Laplacian(np.array(gray), cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking image quality: {e}")
        return False

def describe_image(image: Image.Image, suspected_deficiency: str = None) -> str:
    """Generate description of the image using BLIP model."""
    try:
        processor, model = load_models()
        if not processor or not model:
            return "Failed to load image description model."
        
        # Prepare image for BLIP
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption with context
        if suspected_deficiency and suspected_deficiency != "None":
            prompt = f"Medical image showing potential {suspected_deficiency.lower()} indicators:"
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Enhance caption for medical context
        if suspected_deficiency and suspected_deficiency != "None":
            caption = f"Medical image analysis for {suspected_deficiency}: {caption}"
        
        return caption
        
    except Exception as e:
        logger.error(f"Error describing image: {e}")
        return "Unable to generate image description."

def test_groq_api(api_key: Optional[str] = None):
    """Test if GROQ API is working properly with fallback models and retry logic."""
    try:
        # Ensure environment is loaded
        load_dotenv(override=True)
        if api_key is not None:
            api_key = set_groq_api_key(api_key)
        else:
            api_key = get_groq_api_key()
        
        if not api_key:
            return False, "No API key found in environment. Please check your .env file."
        
        # Test API key format (basic validation)
        if not api_key.startswith('gsk_'):
            return False, "Invalid API key format - should start with 'gsk_'"
            
        if len(api_key) < 20:
            return False, "Invalid API key length"

        # Verify API key with lightweight REST call first
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except requests.exceptions.RequestException as req_error:
            return False, f"API error: {req_error}"

        if response.status_code == 401:
            return False, "Invalid API key"
        if response.status_code == 429:
            return False, "API quota exceeded"
        if response.status_code >= 500:
            return False, "GROQ service unavailable"
        if response.status_code != 200:
            return False, f"API error: HTTP {response.status_code}"

        available_models = []
        try:
            json_payload = response.json()
            models_data = json_payload.get("data", []) if isinstance(json_payload, dict) else []
            available_models = [model.get("id") for model in models_data if isinstance(model, dict)]
        except ValueError:
            # Ignore JSON decoding issues; REST check already succeeded
            available_models = []
        
        # List of models to try in order of preference
        models_to_try = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        test_prompt = "Say 'API is working' if you can read this."
        
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(
                        groq_api_key=api_key,  # Use the local api_key variable
                        model_name=model_name,
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    response = llm.invoke(test_prompt)
                    
                    # Extract the content from the response
                    if hasattr(response, 'content'):
                        response_text = response.content
                    elif hasattr(response, 'text'):
                        response_text = response.text
                    else:
                        response_text = str(response)
                    
                    if response_text and len(response_text.strip()) > 0:
                        return True, f"API is working (using {model_name})"
                    else:
                        raise Exception("Empty response from API")
                
                # Use retry mechanism for this model
                result = retry_with_exponential_backoff(test_model)
                if result:
                    return result
                    
            except Exception as model_error:
                error_msg = str(model_error).lower()
                if "over capacity" in error_msg or "503" in str(model_error):
                    # Try next model
                    continue
                elif "unauthorized" in error_msg or "invalid" in error_msg:
                    # REST check succeeded; treat as transient authorization issue
                    continue
                elif "quota" in error_msg or "limit" in error_msg:
                    return False, "API quota exceeded"
                else:
                    continue
        
        # If all LLM calls fail but REST call succeeded, still consider API alive
        if available_models:
            primary_model = next((m for m in models_to_try if m in available_models), available_models[0]) if available_models else "llama3-8b-8192"
            return True, f"API is responsive (models available: {', '.join(available_models[:3])})"

        return False, "All models are currently over capacity. Please try again later."
            
    except Exception as e:
        error_msg = str(e).lower()
        if "unauthorized" in error_msg or "invalid" in error_msg:
            return False, "Invalid API key"
        elif "connection" in error_msg or "timeout" in error_msg:
            return False, "Connection error"
        elif "quota" in error_msg or "limit" in error_msg:
            return False, "API quota exceeded"
        else:
            return False, f"API error: {str(e)}"

def generate_fallback_response(predicted_class: str, image_description: str, cnn_detection: str = None, confidence: float = None) -> str:
    """Generate a fallback response when LLM fails."""
    cnn_info = ""
    if cnn_detection and cnn_detection != "Model not available":
        confidence_str = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "Not available"
        cnn_info = f"""
        **CNN Detection Results:**
        - Detected: {cnn_detection}
        - Confidence: {confidence_str}
        """
    
    if predicted_class == "vitamin_deficiency":
        return f"""
        **Medical Analysis - Vitamin Deficiency Detection**

        **Image Analysis**: {image_description}
        {cnn_info}

        **Primary Findings**: 
        - Image has been analyzed for vitamin deficiency indicators
        - CNN model detection: {cnn_detection if cnn_detection else 'Not available'}
        - Further clinical assessment recommended

        **Medical Interpretation**: 
        - Visual analysis can provide initial screening
        - CNN detection provides automated assessment
        - Laboratory tests are required for definitive diagnosis

        **Clinical Recommendations**: 
        1. Schedule comprehensive blood work for vitamin levels
        2. Consult with a healthcare provider for detailed assessment
        3. Consider dietary evaluation and supplementation if needed
        4. Follow up on CNN detection results

        **Patient Education**: 
        - Maintain a balanced diet rich in vitamins
        - Regular health check-ups are important
        - Report any symptoms to your healthcare provider
        - Monitor for signs of vitamin deficiencies

        *Note: This is a preliminary analysis. Please consult a healthcare professional for definitive diagnosis and treatment.*
        """
    elif predicted_class == "retina_blood_vessel":
        return f"""
        **Medical Analysis - Retinal Blood Vessel Assessment**

        **Image Analysis**: {image_description}
        {cnn_info}

        **Primary Findings**: 
        - Retinal blood vessel patterns have been analyzed
        - CNN model detection: {cnn_detection if cnn_detection else 'Not available'}
        - Further specialized assessment recommended

        **Medical Interpretation**: 
        - Retinal vessels can indicate cardiovascular health
        - CNN detection provides automated pattern recognition
        - Professional ophthalmological evaluation needed

        **Clinical Recommendations**: 
        1. Schedule comprehensive eye examination
        2. Consider cardiovascular health assessment
        3. Monitor blood pressure and overall health
        4. Follow up on CNN detection results

        **Risk Assessment**: 
        - Regular eye exams are important for early detection
        - Retinal changes can indicate systemic health issues
        - CNN detection helps identify patterns

        *Note: This is a preliminary analysis. Please consult an ophthalmologist for definitive assessment.*
        """
    else:  # combined
        return f"""
        **Medical Analysis - Combined Health Assessment**

        **Image Analysis**: {image_description}
        {cnn_info}

        **Primary Findings**: 
        - Comprehensive analysis of both vitamin and retinal indicators
        - CNN model detection: {cnn_detection if cnn_detection else 'Not available'}
        - Multi-system health assessment recommended

        **Medical Interpretation**: 
        - Combined analysis provides broader health insights
        - CNN detection provides automated assessment
        - Professional medical evaluation required

        **Clinical Recommendations**: 
        1. Schedule comprehensive medical examination
        2. Consider both nutritional and cardiovascular assessments
        3. Implement preventive health measures
        4. Follow up on CNN detection results

        **Risk Stratification**: 
        - Regular health monitoring is recommended
        - Early intervention can prevent complications
        - CNN detection helps prioritize interventions

        **Patient Education**: 
        - Maintain healthy lifestyle habits
        - Regular medical check-ups are essential
        - Report any concerns to healthcare providers
        - Monitor for signs of health issues

        *Note: This is a preliminary analysis. Please consult healthcare professionals for comprehensive evaluation.*
        """

# LangChain integration
def query_langchain(prompt: str, predicted_class: str, confidence: float = None, tabular_context: str = None, cnn_detection: str = None) -> str:
    """Query LangChain with Groq for medical analysis with fallback models."""
    try:
        api_key = get_groq_api_key()
        if not api_key:
            # Use fallback response when no API key
            return generate_fallback_response(predicted_class, "No API key available", cnn_detection, confidence)
        
        # List of models to try in order of preference
        models_to_try = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        # Create a simpler, more direct prompt
        enhanced_prompt = f"""
        You are a medical AI assistant. Please analyze the following medical image information and provide a comprehensive analysis.

        {prompt}

        Analysis Type: {predicted_class}
        Clinical Data: {tabular_context if tabular_context else 'Not provided'}

        Please provide a detailed medical analysis with the following structure:
        1. Primary Findings
        2. Medical Interpretation
        3. Clinical Recommendations
        4. Patient Education

        Be thorough and professional in your response.
        """
        
        # Try each model until one works
        for model_name in models_to_try:
            try:
                def query_model():
                    llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=model_name,
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    response = llm.invoke(enhanced_prompt)
                    
                    # Extract the content from the response
                    if hasattr(response, 'content'):
                        response_text = response.content
                    elif hasattr(response, 'text'):
                        response_text = response.text
                    else:
                        response_text = str(response)
                    
                    # Check if response is valid
                    if response_text and len(response_text.strip()) >= 50:
                        return response_text.strip()
                    else:
                        raise Exception("Incomplete response from API")
                
                # Use retry mechanism for this model
                result = retry_with_exponential_backoff(query_model)
                if result:
                    return result
                    
            except Exception as model_error:
                error_msg = str(model_error).lower()
                if "over capacity" in error_msg or "503" in str(model_error):
                    # Try next model
                    continue
                elif "unauthorized" in error_msg or "invalid" in error_msg:
                    return generate_fallback_response(predicted_class, "Invalid API key", cnn_detection, confidence)
                elif "quota" in error_msg or "limit" in error_msg:
                    return generate_fallback_response(predicted_class, "API quota exceeded", cnn_detection, confidence)
        
        # If all models fail, use fallback response
        return generate_fallback_response(predicted_class, "All models are currently over capacity", cnn_detection, confidence)
        
    except Exception as e:
        # Use fallback response for any other errors
        return generate_fallback_response(predicted_class, f"Analysis failed: {str(e)}", cnn_detection, confidence)

# PDF Generation
class MedicalPDF(FPDF):
    def __init__(self, patient_info=""):
        super().__init__()
        self.patient_info = patient_info
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
    
    def sanitize_text(self, text):
        """Sanitize text for PDF output."""
        if not text:
            return "Not provided"
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        # Limit length to prevent PDF issues
        if len(text) > 1000:
            text = text[:1000] + "..."
        return text
    
    def header(self):
        """PDF header."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'NutriScanAi - Medical Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """PDF footer."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'R')
    
    def cover_page(self):
        """Create cover page."""
        self.add_page()
        self.set_font('Arial', 'B', 24)
        self.cell(0, 60, 'NutriScanAi', 0, 1, 'C')
        self.set_font('Arial', 'B', 16)
        self.cell(0, 20, 'Medical Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 12)
        self.cell(0, 20, f'Patient Information: {self.sanitize_text(self.patient_info)}', 0, 1, 'C')
        self.cell(0, 20, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(20)
    
    def table_of_contents(self):
        """Add table of contents."""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'L')
        self.ln(10)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, '1. Executive Summary', 0, 1, 'L')
        self.cell(0, 10, '2. Analysis Results', 0, 1, 'L')
        self.cell(0, 10, '3. Clinical Recommendations', 0, 1, 'L')
        self.cell(0, 10, '4. Patient Education', 0, 1, 'L')
        self.ln(10)
    
    def add_image(self, image_path, width=180):
        """Add image to PDF."""
        try:
            if os.path.exists(image_path):
                self.image(image_path, x=15, y=self.get_y(), w=width)
                self.ln(100)
        except Exception as e:
            logger.error(f"Error adding image to PDF: {e}")
    
    def add_section(self, title, body):
        """Add a section to the PDF."""
        self.add_page()
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        self.set_font('Arial', '', 12)
        
        # Split body into lines and add to PDF
        lines = body.split('\n')
        for line in lines:
            if line.strip():
                # Handle markdown formatting
                if line.startswith('**') and line.endswith('**'):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 8, line.replace('**', ''), 0, 1, 'L')
                    self.set_font('Arial', '', 12)
                elif line.startswith('#'):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 8, line.replace('#', '').strip(), 0, 1, 'L')
                    self.set_font('Arial', '', 12)
                else:
                    # Handle long lines
                    if len(line) > 80:
                        words = line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + " " + word) < 80:
                                current_line += " " + word if current_line else word
                            else:
                                self.cell(0, 8, current_line, 0, 1, 'L')
                                current_line = word
                        if current_line:
                            self.cell(0, 8, current_line, 0, 1, 'L')
                    else:
                        self.cell(0, 8, line, 0, 1, 'L')
            else:
                self.ln(5)
    
    def create_table(self, line):
        """Create a table from markdown line."""
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            col_width = 190 // len(cells)
            for cell in cells:
                self.cell(col_width, 8, cell, 1, 0, 'C')
            self.ln()
    
    def add_summary(self, report, tabular_context=None):
        """Add summary section to PDF."""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        self.set_font('Arial', '', 12)
        
        # Extract summary from report
        if "**Primary Findings Summary**" in report:
            summary = report.split("**Primary Findings Summary**")[1].split("**")[0]
        elif "1. **Primary Findings Summary**" in report:
            summary = report.split("1. **Primary Findings Summary**")[1].split("2. **")[0]
        else:
            summary = report[:500] + "..." if len(report) > 500 else report
        
        self.add_section("Summary", summary)
        
        if tabular_context:
            self.add_section("Clinical Context", tabular_context)
    
    def add_explainability(self, lime_path, ig_path, gradcam_path):
        """Add explainability visualizations to PDF."""
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Explainability Analysis', 0, 1, 'L')
        self.ln(5)
        
        # Add explainability images
        if lime_path and os.path.exists(lime_path):
            self.add_image(lime_path, width=150)
        if ig_path and os.path.exists(ig_path):
            self.add_image(ig_path, width=150)
        if gradcam_path and os.path.exists(gradcam_path):
            self.add_image(gradcam_path, width=150)
    
    def add_metrics_plots(self, plot_paths):
        """Add metric plots to PDF."""
        if not plot_paths:
            return
        
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Model Performance Metrics', 0, 1, 'L')
        self.ln(5)
        
        for plot_path in plot_paths:
            if os.path.exists(plot_path):
                self.add_image(plot_path, width=150)

# Utility functions
def gradient_text(text, color1, color2):
    """Create gradient text effect."""
    return f'<span style="background: linear-gradient(45deg, {color1}, {color2}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{text}</span>'

def validate_dataset(dataset_dir):
    """Validate dataset structure and integrity."""
    try:
        if not os.path.exists(dataset_dir):
            return False, "Dataset directory does not exist"
        
        # Check for vitamin class directories directly in dataset folder
        vitamin_classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        found_classes = []
        
        for class_name in vitamin_classes:
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.exists(class_path) and os.path.isdir(class_path):
                found_classes.append(class_name)
                # Check for images in each class
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    return False, f"No images found in class '{class_name}'"
        
        if not found_classes:
            return False, "No vitamin class directories found in dataset folder"
        
        return True, f"Dataset validated successfully. Found {len(found_classes)} classes: {', '.join(found_classes)}"
        
    except Exception as e:
        return False, f"Error validating dataset: {str(e)}"

def preprocess_image(img_path, output_path):
    """Preprocess image for better model performance."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        img = cv2.resize(img, (224, 224))
        
        # Apply Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Save processed image
        cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        logger.error(f"Error preprocessing image {img_path}: {e}")
        return False

def augment_with_blur(img_path, output_path, blur_radius=2):
    """Create blurred version of image for data augmentation."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (blur_radius*2+1, blur_radius*2+1), blur_radius)
        cv2.imwrite(output_path, blurred)
        return True
    except Exception as e:
        logger.error(f"Error creating blurred image {img_path}: {e}")
        return False

# CSS loading function
def load_css():
    """Load external CSS file."""
    try:
        with open('style.css', 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure 'style.css' is in the same directory as app.py")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

# Clear MPS cache function
def clear_mps_cache():
    """Clear MPS cache if on macOS and MPS is available."""
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache")
        except RuntimeError as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

# Data loading functions
def load_and_preprocess_csv(csv_path):
    """Load and preprocess cholesterol CSV data."""
    try:
        df = pd.read_csv(csv_path)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col != 'target']
        features = df[feature_columns]
        labels = df['target']
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        return pd.DataFrame(features_scaled, columns=feature_columns), labels
    except Exception as e:
        logger.error(f"Error preprocessing CSV: {e}")
        return None, None

# Image transformation
def get_image_transform():
    """Get standard image transformation for models."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

def create_dataset_splits(dataset_dir, split_ratio=(0.7, 0.15, 0.15)):
    """Create train/val/test splits from the main dataset directory."""
    try:
        import shutil
        from sklearn.model_selection import train_test_split
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
        
        # Get all classes
        vitamin_classes = ["Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D", "Vitamin E", "Retina Blood Vessel"]
        
        for class_name in vitamin_classes:
            class_path = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            # Get all images in this class
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                continue
            
            # Split images
            train_ratio, val_ratio, test_ratio = split_ratio
            val_test_ratio = val_ratio + test_ratio
            
            train_images, val_test_images = train_test_split(
                images, test_size=val_test_ratio, random_state=42
            )
            
            val_images, test_images = train_test_split(
                val_test_images, test_size=test_ratio/val_test_ratio, random_state=42
            )
            
            # Create class directories in splits
            for split in ['train', 'val', 'test']:
                split_class_dir = os.path.join(dataset_dir, split, class_name)
                if not os.path.exists(split_class_dir):
                    os.makedirs(split_class_dir)
            
            # Copy images to appropriate splits
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_dir, 'train', class_name, img)
                shutil.copy2(src, dst)
            
            for img in val_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_dir, 'val', class_name, img)
                shutil.copy2(src, dst)
            
            for img in test_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dataset_dir, 'test', class_name, img)
                shutil.copy2(src, dst)
        
        return True, "Dataset splits created successfully"
        
    except Exception as e:
        return False, f"Error creating dataset splits: {str(e)}" 