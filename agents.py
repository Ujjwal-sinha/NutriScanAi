"""
AI Agents for NutriScanAI - Medical Image Analysis Platform
Enhanced with intelligent agents for comprehensive health analysis
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import time
import random

# LangChain imports
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

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
            print(f"GROQ API over capacity, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(delay)

# Medical knowledge base
MEDICAL_KNOWLEDGE = {
    "vitamin_a": {
        "deficiency_symptoms": ["night blindness", "dry skin", "poor wound healing"],
        "food_sources": ["carrots", "sweet potatoes", "spinach", "liver"],
        "treatments": ["dietary changes", "supplements", "topical treatments"]
    },
    "vitamin_b": {
        "deficiency_symptoms": ["fatigue", "anemia", "neurological issues"],
        "food_sources": ["meat", "fish", "eggs", "dairy", "legumes"],
        "treatments": ["B-complex supplements", "dietary improvements"]
    },
    "vitamin_c": {
        "deficiency_symptoms": ["scurvy", "bleeding gums", "poor wound healing"],
        "food_sources": ["citrus fruits", "bell peppers", "strawberries"],
        "treatments": ["vitamin C supplements", "dietary changes"]
    },
    "vitamin_d": {
        "deficiency_symptoms": ["bone pain", "muscle weakness", "fatigue"],
        "food_sources": ["fatty fish", "egg yolks", "fortified foods"],
        "treatments": ["vitamin D supplements", "sunlight exposure"]
    },
    "vitamin_e": {
        "deficiency_symptoms": ["muscle weakness", "vision problems", "immune issues"],
        "food_sources": ["nuts", "seeds", "vegetable oils", "leafy greens"],
        "treatments": ["vitamin E supplements", "dietary changes"]
    },
    "retina_conditions": {
        "diabetic_retinopathy": {
            "symptoms": ["blurred vision", "floaters", "dark spots"],
            "risk_factors": ["diabetes", "high blood pressure", "smoking"],
            "treatments": ["laser therapy", "injections", "surgery"]
        },
        "hypertensive_retinopathy": {
            "symptoms": ["vision changes", "headaches", "eye pain"],
            "risk_factors": ["high blood pressure", "age", "smoking"],
            "treatments": ["blood pressure control", "medication", "lifestyle changes"]
        }
    }
}

class MedicalImageAnalysisTool(BaseTool):
    name: str = "medical_image_analyzer"
    description: str = "Analyze medical images for vitamin deficiencies and retinal conditions"
    
    def _run(self, image_description: str, detected_condition: str, confidence: float) -> str:
        """Analyze medical image and provide detailed insights"""
        analysis = {
            "condition": detected_condition,
            "confidence": confidence,
            "severity": self._assess_severity(confidence),
            "symptoms": self._get_symptoms(detected_condition),
            "recommendations": self._get_recommendations(detected_condition),
            "risk_level": self._assess_risk(detected_condition, confidence)
        }
        return json.dumps(analysis, indent=2)
    
    def _assess_severity(self, confidence: float) -> str:
        if confidence > 0.9:
            return "High"
        elif confidence > 0.7:
            return "Moderate"
        else:
            return "Low"
    
    def _get_symptoms(self, condition: str) -> List[str]:
        condition_lower = condition.lower()
        for vitamin, data in MEDICAL_KNOWLEDGE.items():
            if vitamin in condition_lower:
                return data.get("deficiency_symptoms", [])
        return ["Consult healthcare professional for specific symptoms"]
    
    def _get_recommendations(self, condition: str) -> List[str]:
        condition_lower = condition.lower()
        for vitamin, data in MEDICAL_KNOWLEDGE.items():
            if vitamin in condition_lower:
                return data.get("treatments", [])
        return ["Seek medical consultation for proper diagnosis and treatment"]

class SymptomCheckerTool(BaseTool):
    name: str = "symptom_checker"
    description: str = "Cross-reference symptoms with detected conditions"
    
    def _run(self, symptoms: str, detected_condition: str) -> str:
        """Check symptoms against detected condition"""
        condition_lower = detected_condition.lower()
        
        # Find matching condition in knowledge base
        for vitamin, data in MEDICAL_KNOWLEDGE.items():
            if vitamin in condition_lower:
                expected_symptoms = data.get("deficiency_symptoms", [])
                symptom_match = self._compare_symptoms(symptoms, expected_symptoms)
                return f"Symptom match: {symptom_match}% - Expected: {expected_symptoms}"
        
        return "Condition not found in knowledge base"

class TreatmentAdvisorTool(BaseTool):
    name: str = "treatment_advisor"
    description: str = "Provide evidence-based treatment recommendations"
    
    def _run(self, condition: str, severity: str) -> str:
        """Provide treatment recommendations"""
        condition_lower = condition.lower()
        
        for vitamin, data in MEDICAL_KNOWLEDGE.items():
            if vitamin in condition_lower:
                treatments = data.get("treatments", [])
                food_sources = data.get("food_sources", [])
                
                return {
                    "treatments": treatments,
                    "food_sources": food_sources,
                    "severity": severity,
                    "urgency": "High" if severity == "High" else "Moderate"
                }
        
        return {"message": "Consult healthcare professional for treatment plan"}

class RiskAssessorTool(BaseTool):
    name: str = "risk_assessor"
    description: str = "Assess health risks based on detected conditions"
    
    def _run(self, condition: str, patient_data: Dict) -> str:
        """Assess health risks"""
        risk_factors = []
        
        # Age-based risks
        age = patient_data.get("age", 30)
        if age > 50:
            risk_factors.append("Age-related risk factors")
        
        # Condition-specific risks
        condition_lower = condition.lower()
        if "retina" in condition_lower:
            risk_factors.extend(["Vision impairment", "Cardiovascular complications"])
        elif any(vitamin in condition_lower for vitamin in ["vitamin_a", "vitamin_d"]):
            risk_factors.append("Immune system compromise")
        
        return {
            "risk_level": "High" if len(risk_factors) > 2 else "Moderate",
            "risk_factors": risk_factors,
            "recommendations": ["Regular monitoring", "Lifestyle modifications"]
        }

class MedicalAIAgent:
    """Main Medical AI Agent for NutriScanAI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = self._get_working_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize tools
        self.tools = [
            MedicalImageAnalysisTool(),
            SymptomCheckerTool(),
            TreatmentAdvisorTool(),
            RiskAssessorTool()
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test")
                    if test_response:
                        return llm
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                return retry_with_exponential_backoff(test_model)
                
            except Exception as e:
                error_msg = str(e).lower()
                if "over capacity" in error_msg or "503" in str(e):
                    continue
                else:
                    # For other errors, try next model
                    continue
        
        # If all models fail, raise an exception
        raise Exception("All GROQ models are currently unavailable")
    
    def analyze_patient_case(self, 
                           image_description: str,
                           detected_condition: str,
                           confidence: float,
                           patient_data: Dict,
                           symptoms: str = "") -> Dict:
        """Comprehensive patient case analysis"""
        
        # Create analysis prompt
        prompt = f"""
        Analyze this medical case comprehensively:
        
        Image Description: {image_description}
        Detected Condition: {detected_condition}
        Confidence: {confidence}
        Patient Data: {patient_data}
        Symptoms: {symptoms}
        
        Provide a detailed analysis including:
        1. Condition assessment
        2. Symptom correlation
        3. Treatment recommendations
        4. Risk assessment
        5. Follow-up recommendations
        """
        
        try:
            response = self.agent.run(prompt)
            return {
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "agent_version": "1.0"
            }
        except Exception as e:
            return {
                "error": str(e),
                "fallback_analysis": self._generate_fallback_analysis(
                    detected_condition, confidence, patient_data
                )
            }
    
    def _generate_fallback_analysis(self, condition: str, confidence: float, patient_data: Dict) -> str:
        """Generate fallback analysis when agent fails"""
        return f"""
        **Fallback Analysis**
        
        Condition: {condition}
        Confidence: {confidence:.1%}
        
        **Recommendations:**
        1. Consult a healthcare professional for proper diagnosis
        2. Schedule follow-up appointments
        3. Monitor symptoms closely
        4. Maintain healthy lifestyle habits
        
        **Note:** This is a preliminary analysis. Professional medical consultation is required.
        """

class ResearchAssistantAgent:
    """Research Assistant for Medical Literature"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = self._get_working_llm()
    
    def _get_working_llm(self):
        """Get a working LLM instance with fallback models and retry logic."""
        models_to_try = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        for model_name in models_to_try:
            try:
                def test_model():
                    llm = ChatGroq(
                        model=model_name,
                        temperature=0.1,
                        groq_api_key=self.api_key
                    )
                    # Test the model with a simple prompt
                    test_response = llm.invoke("Test")
                    if test_response:
                        return llm
                    else:
                        raise Exception("Empty response from model")
                
                # Use retry mechanism for this model
                return retry_with_exponential_backoff(test_model)
                
            except Exception as e:
                error_msg = str(e).lower()
                if "over capacity" in error_msg or "503" in str(e):
                    continue
                else:
                    # For other errors, try next model
                    continue
        
        # If all models fail, raise an exception
        raise Exception("All GROQ models are currently unavailable")
    
    def search_medical_literature(self, condition: str) -> str:
        """Search medical literature for condition"""
        prompt = f"""
        Provide recent medical research findings about: {condition}
        
        Include:
        1. Latest treatment approaches
        2. Clinical guidelines
        3. Risk factors
        4. Prevention strategies
        """
        
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Unable to search literature: {str(e)}"

class DataAnalysisAgent:
    """Data Analysis Agent for Health Trends"""
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_health_trends(self, patient_history: List[Dict]) -> Dict:
        """Analyze health trends from patient history"""
        if not patient_history:
            return {"message": "No patient history available"}
        
        # Analyze trends
        conditions = [entry.get("condition") for entry in patient_history]
        confidences = [entry.get("confidence", 0) for entry in patient_history]
        
        trend_analysis = {
            "total_analyses": len(patient_history),
            "most_common_condition": max(set(conditions), key=conditions.count) if conditions else None,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "trend_direction": "Improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "Stable",
            "recommendations": self._generate_trend_recommendations(conditions, confidences)
        }
        
        return trend_analysis
    
    def _generate_trend_recommendations(self, conditions: List[str], confidences: List[float]) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []
        
        if len(conditions) > 1:
            if conditions[-1] == conditions[-2]:
                recommendations.append("Persistent condition detected - consider specialist consultation")
            
            if confidences[-1] < 0.7:
                recommendations.append("Low confidence in recent analysis - recommend retesting")
        
        recommendations.append("Continue monitoring and regular check-ups")
        return recommendations

# Utility functions
def create_agent_instance(agent_type: str, api_key: str):
    """Create agent instance based on type"""
    if agent_type == "medical":
        return MedicalAIAgent(api_key)
    elif agent_type == "research":
        return ResearchAssistantAgent(api_key)
    elif agent_type == "data":
        return DataAnalysisAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_recommendations(condition: str, patient_data: Dict) -> Dict:
    """Get agent recommendations for a condition"""
    recommendations = {
        "immediate_actions": [],
        "short_term": [],
        "long_term": [],
        "monitoring": []
    }
    
    # Condition-specific recommendations
    condition_lower = condition.lower()
    
    if any(vitamin in condition_lower for vitamin in ["vitamin_a", "vitamin_d", "vitamin_e"]):
        recommendations["immediate_actions"].append("Schedule nutrition consultation")
        recommendations["short_term"].append("Begin dietary modifications")
        recommendations["long_term"].append("Establish healthy eating habits")
    
    if "retina" in condition_lower:
        recommendations["immediate_actions"].append("Schedule ophthalmologist appointment")
        recommendations["short_term"].append("Monitor vision changes")
        recommendations["long_term"].append("Regular eye examinations")
    
    recommendations["monitoring"].extend([
        "Track symptoms daily",
        "Maintain health diary",
        "Regular follow-up appointments"
    ])
    
    return recommendations 