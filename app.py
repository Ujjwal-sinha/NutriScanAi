# Line 1: Import required libraries
import os  # For file system operations
import streamlit as st  # For building the web interface
from PIL import Image, UnidentifiedImageError  # For image processing
from datetime import datetime  # For timestamp in PDF
from dotenv import load_dotenv  # For loading environment variables
import torch  # For PyTorch-based model processing
from transformers import BlipProcessor, BlipForConditionalGeneration  # For image captioning
from langchain_groq import ChatGroq  # For Groq AI integration
from langchain_core.prompts import ChatPromptTemplate  # For prompt templating
from langchain_core.output_parsers import StrOutputParser  # For parsing AI output
from fpdf import FPDF  # For PDF generation
import tempfile  # For temporary file handling
import base64  # For encoding PDF for download

# Line 14: Set Streamlit page configuration
st.set_page_config(
    page_title="NutriScan AI - Vitamin Deficiency Detector",
    layout="centered",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Line 21: Define function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Line 25: Load custom CSS
local_css("style.css")

# Line 27: Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Line 30: Set device for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Line 32: Cache models to prevent reloading
@st.cache_resource
def load_models():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

# Line 42: Load BLIP models
processor, model = load_models()

# Line 44: Define optimized image processing function
def describe_image(image: Image.Image) -> str:
    try:
        if not processor or not model:
            raise ValueError("Models not loaded properly")
        image = image.resize((512, 512))
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=100, num_beams=5)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return ""

# Line 56: Define efficient prompt template
EFFICIENT_PROMPT_TEMPLATE = """
As a board-certified nutritionist with 20+ years experience, analyze this medical image showing: {caption}
Additional patient context: {context}

Generate a comprehensive deficiency analysis with:

1. **Primary Deficiency Identification** (Most likely 1-2 deficiencies)
   - Vitamin/Mineral: 
   - Confidence Level: (High/Medium/Low)
   - Key Visual Indicators: 

2. **Detailed Analysis**
   - Physiological Mechanism: (How deficiency causes these symptoms)
   - Differential Diagnosis: (Other possible explanations)

3. **Evidence-Based Recommendations**
   - Top 5 Food Sources (bioavailable forms)
   - Supplement Protocol (if warranted):
     - Type:
     - Dosage:
     - Duration:
   - Synergistic Nutrients:

4. **Clinical Considerations**
   - Expected Timeline for Improvement
   - When to Consult Physician
   - Red Flag Symptoms

5. **Patient Education**
   - Simple Dietary Changes
   - Preparation Tips
   - Common Pitfalls

Format using markdown with clear headings. Be concise but thorough.
"""

# Line 88: Define optimized LLM query function
def query_langchain(prompt: str) -> str:
    try:
        if not GROQ_API_KEY:
            raise ValueError("API key not found")
        chat = ChatGroq(
            temperature=0.3,
            model_name="llama3-70b-8192",
            groq_api_key=GROQ_API_KEY,
            request_timeout=120,
            max_tokens=4000
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI medical nutrition specialist. Provide accurate, evidence-based analysis."),
            ("user", EFFICIENT_PROMPT_TEMPLATE)
        ])
        chain = prompt_template | chat | StrOutputParser()
        return chain.invoke({"caption": prompt, "context": user_context or "None provided"})
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return ""

# Line 107: Define enhanced PDF generator class
class MedicalPDF(FPDF):
    def __init__(self, patient_info=""):
        super().__init__()
        self.patient_info = patient_info
        self.toc = []
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font('Helvetica', '', 12)
        self.set_author("NutriScan AI")
        self.set_title("NutriScan AI Deficiency Report")
        self.set_subject("Vitamin Deficiency Analysis")
    
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(76, 175, 80)
            self.cell(0, 10, 'NutriScan AI Deficiency Report', 0, 1, 'C')
            self.set_line_width(0.5)
            self.set_draw_color(33, 150, 243)
            self.line(10, 20, 200, 20)
            self.ln(10)
    
    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Page {self.page_no() - 1} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
            self.set_font('Helvetica', 'I', 6)
            self.set_text_color(200, 200, 200)
            self.set_xy(10, 270)
            self.cell(0, 10, 'Powered by NutriScan AI', 0, 0, 'L')
    
    def cover_page(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(76, 175, 80)
        self.cell(0, 20, 'NutriScan AI', 0, 1, 'C')
        self.set_font('Helvetica', '', 16)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'Vitamin Deficiency Analysis Report', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)
        if self.patient_info:
            self.multi_cell(0, 8, f'Patient Information:\n{self.patient_info}')
        self.ln(10)
        self.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 8, 'For educational purposes only', 0, 1, 'C')
    
    def table_of_contents(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(76, 175, 80)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'C')
        self.ln(10)
        self.set_font('Helvetica', '', 12)
        self.set_text_color(0, 0, 0)
        for title, page in self.toc:
            self.cell(0, 8, f'{title} {"." * (50 - len(title))} {page}', ln=1)
        self.ln(10)
    
    def add_image(self, image_path, width=180):
        try:
            self.image(image_path, x=(self.w - width)/2, w=width)
            self.ln(10)
        except Exception as e:
            st.error(f"Failed to add image to PDF: {e}")
    
    def add_section(self, title, body):
        self.toc.append((title, self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        lines = body.split('\n')
        in_list = False
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                if not in_list:
                    in_list = True
                    self.set_left_margin(15)
                self.multi_cell(0, 8, f'• {line[2:]}'.encode('latin1', 'ignore').decode('latin1'))
            elif line.startswith('|'):
                self.set_left_margin(10)
                self.create_table(line)
            else:
                if in_list:
                    in_list = False
                    self.set_left_margin(10)
                self.multi_cell(0, 8, line.encode('latin1', 'ignore').decode('latin1'))
            self.ln(2)
        self.set_left_margin(10)
        self.ln(5)
    
    def create_table(self, line):
        if line.startswith('|'):
            cols = [col.strip() for col in line.split('|')[1:-1]]
            col_width = (self.w - 20) / len(cols)
            self.set_font('Helvetica', 'B', 11)
            for col in cols:
                self.cell(col_width, 10, col.encode('latin1', 'ignore').decode('latin1'), border=1)
            self.ln()
            self.set_font('Helvetica', '', 11)
    
    def add_summary(self, report):
        self.add_page()
        self.toc.append(("Executive Summary", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'Executive Summary', 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        summary = "This report provides an AI-generated analysis of potential vitamin deficiencies based on the provided medical image. Key findings include:\n"
        try:
            primary = report.split("2. **Detailed Analysis**")[0].split("1. **Primary Deficiency Identification**")[1]
            summary += f"- {primary.strip()[:200]}...\n"
            recommendations = report.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0]
            summary += f"- Recommendations: {recommendations.strip()[:200]}...\n"
            summary += "Refer to the detailed sections for comprehensive insights."
        except IndexError:
            summary += "- Unable to generate summary due to report structure. Please review detailed sections."
        self.multi_cell(0, 8, summary.encode('latin1', 'ignore').decode('latin1'))
        self.ln(10)

# Line 185: Define gradient text function for UI
def gradient_text(text, color1, color2):
    return f"""
    <style>
    .gradient-text {{
        background: -webkit-linear-gradient(left, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }}
    </style>
    <div class="gradient-text">{text}</div>
    """

# Line 196: Main UI setup
st.markdown(gradient_text("NutriScan AI", "#4CAF50", "#2196F3"), unsafe_allow_html=True)
st.markdown("### AI-Powered Vitamin Deficiency Detection from Medical Images")

# Line 199: Sidebar content
with st.sidebar:
    st.header("About")
    st.markdown("""
    **NutriScan AI** combines:
    - Computer vision
    - Nutritional biochemistry
    - Clinical medicine
    
    For accurate deficiency detection.
    """)
    st.divider()
    

# Line 211: Image upload with preview
col1, col2 = st.columns([3, 2])
with col1:
    img_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png"],
        help="Clear photos of skin, nails, tongue, or eyes work best"
    )

with col2:
    if img_file:
        try:
            image = Image.open(img_file)
            st.image(image, caption="Preview", use_column_width=True)
        except Exception:
            st.warning("Invalid image file")

# Line 225: Enhanced input form
with st.expander("➕ Additional Clinical Context"):
    user_context = st.text_area(
        "Patient Information",
        placeholder="Age, symptoms duration, medical history, current medications...",
        height=100
    )
    clinical_factors = st.multiselect(
        "Relevant Factors",
        ["Diabetes", "Pregnancy", "Vegan Diet", "Malabsorption", "Alcohol Use"],
        placeholder="Select applicable conditions"
    )

# Line 236: Analysis button with progress
if st.button("Analyze with NutriScan AI", type="primary", use_container_width=True):
    if not img_file:
        st.warning("Please upload an image")
        st.stop()

    with st.spinner("Processing image with AI..."):
        progress_bar = st.progress(0)
        
        try:
            # Step 1: Image processing
            progress_bar.progress(20, text="Analyzing visual patterns")
            image = Image.open(img_file).convert("RGB")
            caption = describe_image(image)
            
            if not caption:
                st.stop()

            # Step 2: Clinical correlation
            progress_bar.progress(50, text="Correlating with clinical data")
            result = query_langchain(caption)
            
            if not result:
                st.stop()

            # Step 3: Display results
            progress_bar.progress(90, text="Formatting report")
            
            with st.container():
                st.subheader("🔬 Comprehensive Deficiency Analysis")
                tab1, tab2, tab3 = st.tabs(["Primary Findings", "Recommendations", "Clinical Notes"])
                
                with tab1:
                    st.markdown(result.split("2. **Detailed Analysis**")[0])
                
                with tab2:
                    if "3. **Evidence-Based Recommendations**" in result:
                        st.markdown(result.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0])
                
                with tab3:
                    if "4. **Clinical Considerations**" in result:
                        st.markdown(result.split("4. **Clinical Considerations**")[1])
            
            st.session_state.report_data = {
                "image": image,
                "report": result,
                "timestamp": datetime.now()
            }
            
            progress_bar.progress(100, text="Analysis complete")
            st.success("✓ Report generated")
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"Analysis failed: {str(e)}")

# Line 279: Enhanced PDF export
if 'report_data' in st.session_state:
    st.divider()
    st.subheader("Report Options")
    
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True):
        with st.spinner("Generating professional report..."):
            try:
                patient_info = st.session_state.get('report_data', {}).get('patient_info', user_context or "Not provided")
                pdf = MedicalPDF(patient_info=patient_info)
                
                pdf.cover_page()
                pdf.add_summary(st.session_state.report_data["report"])
                pdf.table_of_contents()
                
                pdf.add_page()
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    st.session_state.report_data["image"].save(tmp.name, quality=90)
                    pdf.add_image(tmp.name)
                    os.unlink(tmp.name)
                
                report = st.session_state.report_data["report"]
                sections = [
                    ("Clinical Findings", report.split("2. **Detailed Analysis**")[0]),
                    ("Detailed Analysis", report.split("2. **Detailed Analysis**")[1].split("3. **Evidence-Based Recommendations**")[0] if "2. **Detailed Analysis**" in report else ""),
                    ("Treatment Plan", report.split("3. **Evidence-Based Recommendations**")[1] if "3. **Evidence-Based Recommendations**" in report else "")
                ]
                
                for title, body in sections:
                    if body.strip():
                        pdf.add_section(title, body)
                
                pdf_output = pdf.output(dest="S").encode('latin1', 'ignore')
                b64 = base64.b64encode(pdf_output).decode('latin1')
                
                button_html = f"""
                <style>
                    .download-btn {{
                        display: inline-block;
                        padding: 12px 24px;
                        font-size: 16px;
                        font-weight: bold;
                        text-align: center;
                        text-decoration: none;
                        color: white;
                        background: green);
                        border-radius: 8px;
                        border: none;
                        cursor: pointer;
                        transition: background 0.3s ease, transform 0.2s ease;
                        width: 100%;
                        box-sizing: border-box;
                    }}
                    .download-btn:hover {{
                        background: linear-gradient(to right, #45a049, #1e88e5);
                        transform: scale(1.05);
                    }}
                    .download-btn:active {{
                        transform: scale(0.95);
                    }}
                </style>
                <a href="data:application/pdf;base64,{b64}" download="NutriScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf" class="download-btn">
                    📥 Download Your Report
                </a>
                """
                st.markdown(button_html, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")

# Line 322: Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ by <b>Ujjwal Sinha</b> • "
    "<a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)