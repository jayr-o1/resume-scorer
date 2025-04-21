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
    title="Resume Scorer API",
    description="API for analyzing resumes against job descriptions (Public Version)",
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
            
            # Verify the response has all required fields
            if not isinstance(analysis, dict):
                logger.error(f"Invalid analysis result type: {type(analysis)}")
                return get_fallback_response("Invalid analysis result type", resume_sections)
            
            # Check for required fields
            required_fields = ["skills_match", "experience", "education", "certifications", "keywords", "industry"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = {}
            
            # Ensure match_percentage and recommendation are set
            if "match_percentage" not in analysis:
                analysis["match_percentage"] = "0%"
            if "recommendation" not in analysis:
                analysis["recommendation"] = "Error during analysis"
            
            # Add resume sections to the response
            if resume_sections:
                analysis["resume_sections"] = resume_sections
                
            return analysis
            
        except Exception as e:
            # Return a simplified analysis with error but ensuring all required fields are present
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Analysis error: {e}\n{error_trace}")
            
            # Create a valid response with all required fields
            return {
                "error": str(e),
                "match_percentage": "0%",
                "recommendation": "Error during analysis",
                "skills_match": {},  # Empty but valid dictionary
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {},
                "resume_sections": resume_sections or {}
            }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error analyzing resume: {e}\n{error_trace}")
        
        # Ensure we return a valid response format even for exceptions
        return {
            "error": str(e),
            "match_percentage": "0%",
            "recommendation": "Server error during analysis",
            "skills_match": {},
            "experience": {},
            "education": {},
            "certifications": {},
            "keywords": {},
            "industry": {}
        }

# Batch processing endpoint (future implementation)
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
    Process multiple resumes in batch
    
    - **job_summary**: Job summary text
    - **key_duties**: Key duties and responsibilities
    - **essential_skills**: Essential skills required
    - **qualifications**: Required qualifications and experience 
    - **industry_override**: Override for auto-detected industry
    - **resumes**: List of resume PDF files
    """
    try:
        if not resumes:
            return JSONResponse(
                status_code=400,
                content={"error": "No resume files provided"}
            )
        
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
        
        # Create a batch ID
        batch_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        # This would normally save the files and process them in the background
        # For now, return a job ID that could be used for polling
        return {
            "status": "processing",
            "message": f"Processing {len(resumes)} resumes against job requirements",
            "batch_id": f"batch-{batch_id}"
        }
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Skills API - public access
@app.get("/skills")
async def list_skills():
    """List all skills in the ontology"""
    skill_ontology = get_skill_ontology()
    return {"skills": list(skill_ontology.skills_map.keys())}

# Add skill to ontology
@app.post("/skills")
async def add_skill(
    skill: SkillRequest
):
    """
    Add a new skill to the ontology (public for demo purposes)
    
    - **name**: Name of the skill
    - **aliases**: Alternative names
    - **category**: Category of the skill 
    - **related**: Related skills
    """
    skill_ontology = get_skill_ontology()
    success = skill_ontology.add_skill(
        skill.name,
        skill.aliases,
        skill.category,
        skill.related
    )
    
    if not success:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to add skill to the ontology"}
        )
    
    return {"status": "success", "message": f"Added skill: {skill.name}"}

# Public debug endpoint for PDF extraction
@app.post("/debug/extract")
async def debug_extract_public(
    resume: UploadFile = File(...)
):
    """
    Debug endpoint to extract text and sections from a PDF resume (public version)
    
    - **resume**: PDF resume file
    """
    try:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await resume.read())
            temp_file_path = temp_file.name
        
        # Extract text with enhanced methods
        extraction_result = extract_text_from_pdf(temp_file_path)
        
        # Also extract sections
        sections = {}
        if extraction_result and extraction_result.get("text"):
            sections = extract_resume_sections(extraction_result["text"])
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "full_result": extraction_result,
            "sections": sections
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error extracting PDF: {e}\n{error_trace}")
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 