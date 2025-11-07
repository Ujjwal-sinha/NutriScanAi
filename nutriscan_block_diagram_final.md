# NutriScanAI - Complete Architecture Block Diagram

## High-Level System Architecture (10 Main Blocks)

```mermaid
graph TB
    %% User Interface Block
    subgraph "1. User Interface"
        UI[Streamlit Web App\nImage Upload & Cropping\nAnalysis Settings\nResults Display]
    end

    %% Input Processing Block
    subgraph "2. Input Processing"
        IP[Image Preprocessing\nQuality Check\nEye Detection\nData Validation]
    end

    %% AI Models Block
    subgraph "3. Fine-tuned AI Models"
        AI[Fine-tuned CNN Model\nFine-tuned MobileNet V2\nFine-tuned MLP Model\nFine-tuned BLIP Model\nCombined Analysis]
    end

    %% AI Agents Block
    subgraph "4. AI Agents"
        AG[Medical AI Agent\nResearch Assistant\nData Analysis Agent\nTool Integration]
    end

    %% Analysis Tools Block
    subgraph "5. Analysis Tools"
        AT[Medical Image Analysis\nSymptom Checker\nTreatment Advisor\nRisk Assessor]
    end

    %% Explainability Block
    subgraph "6. Explainability"
        EX[LIME Analysis\nGrad-CAM\nIntegrated Gradients\nSHAP Analysis]
    end

    %% Data Management Block
    subgraph "7. Data Management"
        DM[Dataset Management\nModel Storage\nCache System\nTemporary Files]
    end

    %% External Services Block
    subgraph "8. External Services"
        ES[GROQ API\nLangChain Framework\nOpenCV Libraries\nPyTorch Framework]
    end

    %% Output Generation Block
    subgraph "9. Output Generation"
        OG[Medical Reports\nPDF Generation\nPerformance Metrics\nData Visualization]
    end

    %% Security & Compliance Block
    subgraph "10. Security & Compliance"
        SC[Data Privacy\nHIPAA Compliance\nError Handling\nFallback Mechanisms]
    end

    %% Main Flow Connections
    UI --> IP
    IP --> AI
    AI --> AG
    AG --> AT
    AT --> EX
    EX --> OG
    
    %% Supporting Connections
    DM --> AI
    ES --> AG
    ES --> AT
    SC --> IP
    SC --> OG
    
    %% Cross-connections
    AI --> EX
    AG --> EX
    AT --> OG
    DM --> OG

    %% Styling
    classDef userBlock fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef processingBlock fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef modelBlock fill:#e8f5e8,stroke:#1b5e20,stroke-width:3px
    classDef agentBlock fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef toolBlock fill:#fce4ec,stroke:#880e4f,stroke-width:3px
    classDef explainBlock fill:#f1f8e9,stroke:#33691e,stroke-width:3px
    classDef dataBlock fill:#fafafa,stroke:#424242,stroke-width:3px
    classDef serviceBlock fill:#e0f2f1,stroke:#004d40,stroke-width:3px
    classDef outputBlock fill:#fff8e1,stroke:#f57f17,stroke-width:3px
    classDef securityBlock fill:#ffebee,stroke:#c62828,stroke-width:3px

    class UI userBlock
    class IP processingBlock
    class AI modelBlock
    class AG agentBlock
    class AT toolBlock
    class EX explainBlock
    class DM dataBlock
    class ES serviceBlock
    class OG outputBlock
    class SC securityBlock
```

## Block Descriptions

### 1. User Interface
- **Streamlit Web App**: Main application interface
- **Image Upload & Cropping**: Interactive image handling
- **Analysis Settings**: User configuration options
- **Results Display**: Comprehensive results presentation

### 2. Input Processing
- **Image Preprocessing**: Quality enhancement and normalization
- **Quality Check**: Validation of image suitability
- **Eye Detection**: Automated eye region identification
- **Data Validation**: Input verification and error handling

### 3. Fine-tuned AI Models
- **Fine-tuned CNN Model**: MobileNet V2 for vitamin deficiency classification
- **Fine-tuned MLP Model**: Cholesterol analysis and numerical processing
- **Fine-tuned BLIP Model**: Image captioning and description generation
- **Combined Analysis**: Integrated model predictions

### 4. AI Agents
- **Medical AI Agent**: Primary medical analysis and diagnosis
- **Research Assistant**: Literature search and evidence integration
- **Data Analysis Agent**: Trend analysis and pattern recognition
- **Tool Integration**: Coordination of specialized tools

### 5. Analysis Tools
- **Medical Image Analysis**: Condition severity assessment
- **Symptom Checker**: Symptom correlation analysis
- **Treatment Advisor**: Evidence-based recommendations
- **Risk Assessor**: Health risk evaluation

### 6. Explainability
- **LIME Analysis**: Local interpretable model explanations
- **Grad-CAM**: Gradient-weighted class activation mapping
- **Integrated Gradients**: Attribution analysis
- **SHAP Analysis**: SHapley Additive exPlanations

### 7. Data Management
- **Dataset Management**: Training/validation/test data organization
- **Model Storage**: Pre-trained model weights and configurations
- **Cache System**: Performance optimization and caching
- **Temporary Files**: Processing file management

### 8. External Services
- **GROQ API**: Large language model services
- **LangChain Framework**: Agent orchestration and tool management
- **OpenCV Libraries**: Computer vision operations
- **PyTorch Framework**: Deep learning computations

### 9. Output Generation
- **Medical Reports**: Comprehensive analysis documentation
- **PDF Generation**: Professional report creation
- **Performance Metrics**: Model evaluation and statistics
- **Data Visualization**: Interactive charts and graphs

### 10. Security & Compliance
- **Data Privacy**: Local processing and privacy protection
- **HIPAA Compliance**: Medical data handling standards
- **Error Handling**: Robust error management and recovery
- **Fallback Mechanisms**: Alternative processing paths

## System Flow

1. **User Interface** receives medical images and user inputs
2. **Input Processing** validates and prepares the data
3. **Fine-tuned AI Models** perform initial analysis and classification
4. **AI Agents** coordinate comprehensive medical analysis
5. **Analysis Tools** provide specialized medical insights
6. **Explainability** ensures transparency in AI decisions
7. **Data Management** supports all processing operations
8. **External Services** provide additional AI capabilities
9. **Output Generation** creates professional medical reports
10. **Security & Compliance** ensures data protection throughout

## Key Features

- **10 Main Blocks**: Simplified architecture overview
- **Clear Data Flow**: Logical progression from input to output
- **Comprehensive Coverage**: All major system components included
- **Professional Design**: Clean, medical-themed color scheme
- **Scalable Structure**: Modular design for easy extension 