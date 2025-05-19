# NutriScan AI - Vitamin Deficiency Detector

## Overview
NutriScan AI is a cutting-edge web application designed to analyze medical images for signs of vitamin and mineral deficiencies using advanced AI models. By combining image captioning with large language model analysis, it generates detailed, evidence-based nutritional deficiency reports tailored to patient context.

---

## Features

- **Image Captioning**: Utilizes Salesforce's BLIP model to generate descriptive captions from uploaded medical images.
- **AI Nutritional Analysis**: Employs Groq AI's LLaMA 3 model via LangChain to produce comprehensive deficiency assessments.
- **Structured Reports**: Includes primary deficiency identification, physiological mechanisms, differential diagnosis, dietary and supplement recommendations, clinical considerations, and patient education.
- **PDF Report Generation**: Creates professional multi-page PDF reports with cover page, table of contents, images, and formatted sections.
- **Interactive Web UI**: Built with Streamlit featuring custom CSS for a clean, user-friendly interface.
- **Secure API Key Management**: Supports environment variables for safe Groq API key handling.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU recommended for performance
- Groq API key

### Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:


4. Create a `.env` file with your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

---

## Usage

Run the Streamlit app:


- Upload a medical image (e.g., showing skin or nail symptoms).
- Optionally provide additional patient context.
- View the AI-generated deficiency analysis.
- Download a detailed PDF report.

---

## Code Structure

- **app.py**: Main application script handling UI, model loading, image captioning, AI querying, and PDF generation.
- **style.css**: Custom CSS styling for the Streamlit interface.
- **requirements.txt**: Lists all Python dependencies with exact versions.

---

## Detailed Report Contents

1. **Primary Deficiency Identification**  
   - Most likely 1-2 vitamin/mineral deficiencies  
   - Confidence level (High/Medium/Low)  
   - Key visual indicators from the image  

2. **Detailed Analysis**  
   - Physiological mechanisms causing symptoms  
   - Differential diagnosis  

3. **Evidence-Based Recommendations**  
   - Top 5 bioavailable food sources  
   - Supplement protocols (type, dosage, duration, synergistic nutrients)  

4. **Clinical Considerations**  
   - Expected timeline for improvement  
   - When to consult a physician  
   - Red flag symptoms  

5. **Patient Education**  
   - Simple dietary changes  
   - Preparation tips  
   - Common pitfalls  

---

## PDF Generation

The `MedicalPDF` class extends `FPDF` to produce:

- Cover page with patient info and generation timestamp
- Table of contents with clickable sections
- Embedded images centered on the page
- Well-formatted text sections with headings, lists, and tables
- Headers and footers with branding and page numbers

---

## Customization & Development

- Modular code with caching for model loading to improve performance.
- Easily extendable prompt templates and UI components.
- Contributions welcome via GitHub pull requests.

---

## License & Disclaimer

For educational and research use only. Not a substitute for professional medical advice.

---

## Contact

Developed by **Ujjwal Sinha**  
For support or inquiries, please open an issue on the project repository.

---

This README provides a comprehensive guide to install, run, and understand the NutriScan AI project for vitamin deficiency detection from medical images.
