# 🔬 NutriScanAi - AI-Powered Medical Analysis Tool

## 📋 Overview

NutriScanAi is an advanced AI-powered medical analysis tool that combines computer vision, nutritional biochemistry, and clinical medicine to detect vitamin deficiencies and analyze retina blood vessel patterns. The application provides comprehensive health assessments with explainable AI techniques.

## 🏗️ Project Structure

The application has been modularized into three main files for better maintainability and understanding:

### 📁 File Organization

```
NutriScanAi/
├── app.py              # Main Streamlit application (1700+ lines)
├── models.py           # ML models and training functions (800+ lines)
├── utils.py            # Utility functions and helpers (400+ lines)
├── agents.py           # AI agents system (361 lines)
├── style.css           # Custom CSS styling
├── requirements.txt    # Python dependencies
├── README.md          # This documentation
└── dataset/           # Training data
    ├── train/         # Training images
    ├── val/           # Validation images
    ├── test/          # Test images
    └── dataset_2190_cholesterol.csv
```

## 🔧 Module Breakdown

### 1. **`app.py`** - Main Application (1700+ lines)
**Purpose**: Main Streamlit interface and application logic

**Key Components**:
- Streamlit page configuration and UI setup
- Sidebar with model configuration and training controls
- Image input handling (upload/camera)
- Clinical context input forms
- Analysis type selection (checkboxes)
- Report generation and PDF export
- Session state management

**Main Features**:
- ✅ Clean, modular UI with external CSS
- ✅ Multiple analysis type selection
- ✅ Real-time image processing
- ✅ Clinical data integration
- ✅ PDF report generation

### 2. **`models.py`** - Machine Learning Models (800+ lines)
**Purpose**: All ML-related functionality

**Key Components**:
- **Dataset Classes**: `VitaminDataset` for image data
- **Model Architectures**: 
  - CNN (MobileNetV2) for image classification
  - MLP for cholesterol prediction
- **Training Functions**: 
  - `train_model()` for CNN training
  - `train_mlp_model()` for MLP training
- **Evaluation Functions**: 
  - `evaluate_combined_model()` for comprehensive evaluation
- **Explainability**: 
  - `apply_lime()` - LIME explanations
  - `apply_integrated_gradients()` - Integrated Gradients
  - `apply_gradcam()` - Grad-CAM visualizations
- **Plotting**: `plot_metrics()` for training curves and performance metrics
- **Evaluation Dashboard**: `create_evaluation_dashboard()` for comprehensive model evaluation visualizations

**Supported Analysis Types**:
- 🔍 **Vitamin Deficiency Detection** (A, B, C, D, E)
- 🩸 **Retina Blood Vessel Analysis**
- 🔬 **Combined Analysis**

### 3. **`utils.py`** - Utility Functions (400+ lines)
**Purpose**: Helper functions and utilities

**Key Components**:
- **Image Processing**: 
  - `check_image_quality()` - Image quality validation
  - `describe_image()` - BLIP-based image captioning
  - `preprocess_image()` - Image preprocessing
- **AI Integration**: 
  - `query_langchain()` - LangChain with Groq API
  - `load_models()` - BLIP model loading
- **PDF Generation**: 
  - `MedicalPDF` class for professional reports
- **Data Validation**: 
  - `validate_dataset()` - Dataset integrity checks
- **CSS Loading**: `load_css()` - External styling
- **Device Management**: MPS cache clearing for Apple Silicon

### 4. **`agents.py`** - AI Agents System (361 lines)
**Purpose**: Intelligent AI agents for comprehensive medical analysis

**Key Components**:
- **MedicalAIAgent**: Main medical analysis agent with multiple specialized tools
- **ResearchAssistantAgent**: Medical literature research and evidence-based insights
- **DataAnalysisAgent**: Health trend analysis and pattern recognition
- **Specialized Tools**: Medical image analysis, symptom checking, treatment advising, and risk assessment

## 🚀 Features

### 🔍 **Analysis Capabilities**
- **Vitamin Deficiency Detection**: Analyze skin, nail, and tongue images
- **Retina Blood Vessel Analysis**: Detect cardiovascular indicators
- **Combined Assessment**: Comprehensive health evaluation
- **Cholesterol Prediction**: MLP-based risk assessment
- **Multi-Agent Analysis**: Intelligent AI agents for comprehensive medical insights

### 🎨 **User Interface**
- **Modern Design**: Clean white background with blue accents
- **Evaluation Dashboard**: Comprehensive model performance analysis with separate tabs

### 📊 **Model Evaluation Dashboard**
The application now includes a comprehensive evaluation dashboard that appears after model training, featuring:

- **📈 ROC Curves**: Receiver Operating Characteristic curves showing model performance across different classes with AUC scores
- **📋 Precision Table**: Detailed performance metrics table with precision, recall, F1-score, and AUC for each class
- **📊 Accuracy Graph**: Training and validation accuracy/loss curves showing model learning progress
- **🎯 Confusion Matrix**: Raw and normalized confusion matrices for detailed prediction analysis
- **📈 Performance Summary**: Overall model performance metrics including accuracy, macro precision, macro recall, and macro F1-score

Each visualization includes detailed explanations to help users understand the metrics and their significance.
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Elements**: Real-time image cropping and analysis
- **Progress Indicators**: Visual feedback during processing

### 🤖 **AI Agents System**
The application features a sophisticated multi-agent system for comprehensive medical analysis:

#### **🏥 MedicalAIAgent - Main Medical Analysis Agent**
- **Primary Role**: Comprehensive patient case analysis and medical decision support
- **Capabilities**:
  - Analyzes medical images with detected conditions and confidence levels
  - Correlates symptoms with detected conditions
  - Provides evidence-based treatment recommendations
  - Assesses health risks based on patient data
  - Generates comprehensive medical reports
- **Tools**: Medical image analysis, symptom checking, treatment advising, risk assessment
- **Output**: Detailed analysis with condition assessment, treatment plans, and follow-up recommendations

#### **📚 ResearchAssistantAgent - Medical Literature Expert**
- **Primary Role**: Medical research and evidence-based insights
- **Capabilities**:
  - Searches medical literature for latest treatment approaches
  - Provides clinical guidelines and best practices
  - Identifies risk factors and prevention strategies
  - Offers evidence-based medical recommendations
- **Focus**: Vitamin deficiencies, retinal conditions, and related medical conditions
- **Output**: Research-backed medical insights and treatment protocols

#### **📊 DataAnalysisAgent - Health Trend Analyst**
- **Primary Role**: Pattern recognition and health trend analysis
- **Capabilities**:
  - Analyzes patient history for health trends
  - Identifies patterns in condition progression
  - Tracks confidence levels over time
  - Provides trend-based recommendations
- **Features**:
  - Trend direction analysis (improving/stable/declining)
  - Most common condition identification
  - Average confidence tracking
  - Pattern-based health recommendations

#### **🔧 Specialized Analysis Tools**
- **MedicalImageAnalysisTool**: Analyzes medical images for vitamin deficiencies and retinal conditions
- **SymptomCheckerTool**: Cross-references symptoms with detected conditions
- **TreatmentAdvisorTool**: Provides evidence-based treatment recommendations
- **RiskAssessorTool**: Assesses health risks based on detected conditions and patient data

### 📊 **AI Explainability**
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Integrated Gradients**: Pixel importance visualization
- **Grad-CAM**: Class activation mapping
- **Performance Metrics**: Confusion matrices, ROC curves, precision-recall

### 📄 **Report Generation**
- **Professional PDFs**: Comprehensive medical reports
- **Multiple Sections**: Findings, recommendations, education
- **Visual Elements**: Charts, graphs, and explainability plots
- **Customizable**: Different formats for different analysis types

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
1. **Create `.env` file**:
```env
GROQ_API_KEY=your_groq_api_key_here
```

2. **Dataset Structure**:
```
dataset/
├── train/
│   ├── Vitamin A/
│   ├── Vitamin B/
│   ├── Vitamin C/
│   ├── Vitamin D/
│   ├── Vitamin E/
│   └── Retina Blood Vessel/
├── val/
├── test/
└── dataset_2190_cholesterol.csv
```

### Running the Application
```bash
streamlit run app.py
```

## 📈 Usage Guide

### 1. **Model Training**
- Use sidebar controls to configure training parameters
- Click "Train and Evaluate Models" to start training
- Monitor training progress and performance metrics

### 2. **Image Analysis**
- Upload image or capture from camera
- Crop image if needed
- Select analysis type(s) using checkboxes
- Click "Start Analysis" to begin processing

### 3. **Clinical Context**
- Enter patient information
- Provide suspected deficiency (optional)
- Add clinical notes and symptoms

### 4. **Report Generation**
- After analysis, generate comprehensive PDF report
- Download report for medical records
- Share with healthcare professionals

## 🔧 Technical Details

### **Model Architecture**
- **CNN**: MobileNetV2 with custom classifier (6 classes)
- **MLP**: 4-layer neural network for cholesterol prediction
- **Image Processing**: 224x224 resolution, ImageNet normalization
- **Training**: Adam optimizer, CrossEntropyLoss, early stopping

### **AI Integration**
- **BLIP**: Image captioning and description
- **LangChain**: Natural language analysis and agent framework
- **Groq API**: Fast LLM inference for AI agents
- **Multi-Agent System**: MedicalAIAgent, ResearchAssistantAgent, DataAnalysisAgent
- **Explainability**: LIME, Integrated Gradients, Grad-CAM

### **Performance Optimization**
- **Apple Silicon**: MPS acceleration support
- **Caching**: Streamlit resource caching
- **Memory Management**: Automatic cache clearing
- **Batch Processing**: Efficient data loading

## 🎯 Benefits of Modularization

### ✅ **Maintainability**
- **Separation of Concerns**: Each file has a specific purpose
- **Easy Debugging**: Isolated functionality for troubleshooting
- **Code Reusability**: Functions can be imported and reused

### ✅ **Scalability**
- **Independent Development**: Teams can work on different modules
- **Easy Testing**: Unit tests for individual components
- **Feature Addition**: New features can be added without affecting others

### ✅ **Understanding**
- **Clear Structure**: Easy to understand project organization
- **Reduced Complexity**: Smaller, focused files
- **Better Documentation**: Each module can be documented separately

## 🔮 Future Enhancements

### **Planned Features**
- [ ] Additional vitamin deficiency types
- [ ] Real-time video analysis
- [ ] Mobile app version
- [ ] Integration with EHR systems
- [ ] Multi-language support

### **Technical Improvements**
- [ ] Model quantization for faster inference
- [ ] Cloud deployment options
- [ ] API endpoints for integration
- [ ] Advanced explainability techniques

## 📞 Support & Contact

- **Documentation**: This README and inline code comments
- **Issues**: Report bugs and feature requests
- **Contributions**: Welcome community contributions
- **LinkedIn**: Connect with the development team

## 📄 License

This project is for educational and research purposes. Please consult healthcare professionals for medical advice.

---

**🔬 NutriScanAi** - Empowering healthcare with AI-driven insights
