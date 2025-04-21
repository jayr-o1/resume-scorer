import sys
import os
import time
import tempfile
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Add the src directory to the Python path
current_dir = Path(__file__).parent.parent
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the FastAPI app - create a custom version of the app
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
from src.utils.pdf_extractor import extract_text_from_pdf, extract_resume_sections
from src.utils.analyzer import analyze_resume, batch_process_resumes
from src.utils.skill_ontology import get_skill_ontology
from src.utils.visualizations import (
    create_skill_radar, 
    create_comparison_chart,
    create_missing_skills_chart,
    create_detailed_skills_breakdown
)

# Create the app
app = FastAPI(
    title="Resume Scorer API - Local",
    description="API for analyzing resumes against job descriptions (Local Development Version)",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class JobDetails(BaseModel):
    summary: Optional[str] = Field(None, description="Job summary text")
    duties: Optional[str] = Field(None, description="Key duties and responsibilities")
    skills: Optional[str] = Field(None, description="Essential skills required")
    qualifications: Optional[str] = Field(None, description="Required qualifications and experience")
    industry_override: Optional[str] = Field(None, description="Override for auto-detected industry")

class ResumeAnalysisResponse(BaseModel):
    match_percentage: str
    recommendation: str
    skills_match: Dict
    experience: Dict
    education: Dict
    certifications: Dict
    keywords: Dict
    industry: Dict
    benchmark: Optional[Dict] = None
    improvement_suggestions: Optional[Dict] = None
    confidence_scores: Optional[Dict] = None
    salary_estimate: Optional[Dict] = None
    error: Optional[str] = None
    resume_sections: Optional[Dict] = None

class BatchAnalysisRequest(BaseModel):
    job_summary: Optional[str] = None
    key_duties: Optional[str] = None
    essential_skills: Optional[str] = None 
    qualifications: Optional[str] = None
    industry_override: Optional[str] = None
    resume_files: List[str] = []

class SkillRequest(BaseModel):
    name: str
    aliases: Optional[List[str]] = None
    category: Optional[str] = None
    related: Optional[List[str]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "environment": "local",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Define a fallback function to ensure valid response structure
def get_fallback_response(error_message, resume_sections=None):
    """Create a fallback response when analysis fails"""
    return {
        "error": str(error_message),
        "match_percentage": "0%",
        "recommendation": "Error during analysis",
        "skills_match": {},
        "experience": {},
        "education": {},
        "certifications": {},
        "keywords": {},
        "industry": {},
        "resume_sections": resume_sections or {}
    }

# Analyze resume endpoint - public version matching the Streamlit app
@app.post("/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume_endpoint(
    resume: UploadFile = File(...),
    job_summary: str = Form(None),
    key_duties: str = Form(None),
    essential_skills: str = Form(None),
    qualifications: str = Form(None),
    industry_override: str = Form(None),
    translate: bool = Form(False)
):
    """
    Analyze a resume against job details
    
    - **resume**: PDF resume file
    - **job_summary**: Job summary text 
    - **key_duties**: Key duties and responsibilities
    - **essential_skills**: Essential skills required
    - **qualifications**: Required qualifications and experience
    - **industry_override**: Override for auto-detected industry
    - **translate**: Whether to translate non-English resumes to English
    """
    try:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await resume.read())
            temp_file_path = temp_file.name
        
        # Extract text from PDF with enhanced methods
        extraction_result = extract_text_from_pdf(temp_file_path, translate=translate)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        logger.info(f"Extraction result for {resume.filename}: {len(extraction_result.get('text', '')) if extraction_result else 'None'} characters")
        
        if not extraction_result:
            return {
                "error": "Could not extract text from the PDF", 
                "match_percentage": "0%",
                "recommendation": "Please provide a valid PDF resume",
                "skills_match": {},
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {}
            }
            
        if not extraction_result.get("text"):
            # Fix the case where text is None - convert to empty string
            extraction_result["text"] = ""
        
        # Convert job details to dict
        job_dict = {
            "summary": job_summary or "",
            "duties": key_duties or "",
            "skills": essential_skills or "",
            "qualifications": qualifications or ""
        }
        
        # Add industry override if provided
        if industry_override and industry_override != "Auto-detect":
            job_dict["industry_override"] = industry_override
        
        # Ensure job details are not empty
        if not any(value for value in job_dict.values() if value):
            job_dict = {
                "summary": "Not provided",
                "skills": "Not specified"
            }
        
        # Get resume sections first for backup
        resume_sections = extract_resume_sections(extraction_result["text"])
        
        # Analyze the resume
        try:
            analysis = analyze_resume(extraction_result, job_dict)
            
            # Validate the response structure before returning
            required_fields = ["match_percentage", "recommendation", "skills_match", 
                              "experience", "education", "certifications", "keywords", "industry"]
            
            # Ensure all required fields exist
            for field in required_fields:
                if field not in analysis:
                    if field == "match_percentage":
                        analysis[field] = "0%"
                    else:
                        analysis[field] = {}
            
            # Ensure match_percentage is a string with % symbol
            if not isinstance(analysis["match_percentage"], str):
                analysis["match_percentage"] = f"{analysis['match_percentage']}%"
            elif not analysis["match_percentage"].endswith("%"):
                analysis["match_percentage"] = f"{analysis['match_percentage']}%"
            
            # Add resume sections to the response
            if resume_sections and not analysis.get("resume_sections"):
                analysis["resume_sections"] = resume_sections
                
            return analysis
        except Exception as e:
            logger.error(f"Error during resume analysis: {str(e)}")
            return get_fallback_response(f"Analysis error: {str(e)}", resume_sections)
        
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return get_fallback_response(f"Processing error: {str(e)}")

# Batch analyze endpoint
@app.post("/batch-analyze")
async def batch_analyze_endpoint(
    background_tasks: BackgroundTasks,
    job_summary: str = Form(None),
    key_duties: str = Form(None),
    essential_skills: str = Form(None),
    qualifications: str = Form(None),
    industry_override: str = Form(None),
    resumes: List[UploadFile] = File(...)
):
    """
    Batch analyze multiple resumes against the same job description
    
    - **job_summary**: Job summary text
    - **key_duties**: Key duties and responsibilities 
    - **essential_skills**: Essential skills required
    - **qualifications**: Required qualifications and experience
    - **industry_override**: Override for auto-detected industry
    - **resumes**: List of PDF resume files to analyze
    """
    try:
        # Convert job details to dict
        job_dict = {
            "summary": job_summary or "",
            "duties": key_duties or "",
            "skills": essential_skills or "",
            "qualifications": qualifications or ""
        }
        
        # Add industry override if provided
        if industry_override and industry_override != "Auto-detect":
            job_dict["industry_override"] = industry_override
            
        # Save the uploaded files
        temp_files = []
        for resume in resumes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(await resume.read())
                temp_files.append((resume.filename, temp_file.name))
        
        # Run analysis in background task
        results = {}
        for filename, file_path in temp_files:
            # Extract text from PDF
            extraction_result = extract_text_from_pdf(file_path)
            
            # Analyze the resume
            if extraction_result and extraction_result.get("text"):
                try:
                    analysis = analyze_resume(extraction_result, job_dict)
                    
                    # Validate the response structure
                    required_fields = ["match_percentage", "recommendation", "skills_match", 
                                     "experience", "education", "certifications", "keywords", "industry"]
                    
                    # Ensure all required fields exist
                    for field in required_fields:
                        if field not in analysis:
                            if field == "match_percentage":
                                analysis[field] = "0%"
                            else:
                                analysis[field] = {}
                    
                    # Ensure match_percentage is a string with % symbol
                    if not isinstance(analysis["match_percentage"], str):
                        analysis["match_percentage"] = f"{analysis['match_percentage']}%"
                    elif not analysis["match_percentage"].endswith("%"):
                        analysis["match_percentage"] = f"{analysis['match_percentage']}%"
                    
                    # Extract resume sections
                    resume_sections = extract_resume_sections(extraction_result["text"])
                    if resume_sections and not analysis.get("resume_sections"):
                        analysis["resume_sections"] = resume_sections
                    
                    results[filename] = analysis
                except Exception as e:
                    logger.error(f"Error analyzing resume {filename}: {str(e)}")
                    results[filename] = get_fallback_response(f"Analysis error: {str(e)}")
            else:
                results[filename] = get_fallback_response("Could not extract text from PDF")
        
        # Clean up temporary files
        for _, file_path in temp_files:
            os.unlink(file_path)
            
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        return {"error": str(e)}

# Skills ontology endpoint
@app.get("/skills")
async def list_skills():
    """Get the entire skill ontology"""
    skill_ontology = get_skill_ontology()
    return {"skills": skill_ontology}

# Add a new skill
@app.post("/skills")
async def add_skill(
    skill: SkillRequest
):
    """
    Add a new skill to the ontology (for local development only)
    
    - **name**: Name of the skill
    - **aliases**: Optional list of alternative names for the skill
    - **category**: Optional category the skill belongs to
    - **related**: Optional list of related skills
    """
    try:
        # Get existing skill ontology
        skill_ontology = get_skill_ontology()
        
        # Add new skill
        skill_data = {
            "name": skill.name,
            "aliases": skill.aliases or [],
            "category": skill.category or "other",
            "related": skill.related or []
        }
        
        # Check if skill already exists
        for existing_skill in skill_ontology:
            if existing_skill["name"].lower() == skill.name.lower():
                return {"error": "Skill already exists"}
        
        # Add to ontology
        skill_ontology.append(skill_data)
        
        # Save back to file (in-memory only for this example)
        
        return {"success": True, "skill": skill_data}
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint for extracting text from a PDF
@app.post("/debug/extract")
async def debug_extract_public(
    resume: UploadFile = File(...)
):
    """Debug endpoint to extract text from a PDF"""
    try:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await resume.read())
            temp_file_path = temp_file.name
        
        # Extract text from PDF
        extraction_result = extract_text_from_pdf(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # Extract sections
        resume_sections = extract_resume_sections(extraction_result["text"])
        
        # Return the extracted text and metadata
        return {
            "text": extraction_result["text"],
            "sections": resume_sections,
            "metadata": extraction_result.get("metadata", {})
        }
    except Exception as e:
        return {"error": str(e)} 