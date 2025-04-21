"""
FastAPI wrapper for the Resume Scorer system
"""

import os
import time
import tempfile
import json
import hashlib
import logging
import uvicorn
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache
import traceback

import jwt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, Header, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.pdf_extractor import extract_text_from_pdf, extract_resume_sections
from utils.analyzer import analyze_resume, batch_process_resumes
from utils.skill_ontology import get_skill_ontology
from utils.visualizations import (
    create_skill_radar, 
    create_comparison_chart,
    create_missing_skills_chart,
    create_detailed_skills_breakdown
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file if available
from dotenv import load_dotenv
load_dotenv()

# Config
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "fallback_secret_key")  # Fallback for local development
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Rate limiting settings
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = {
    "free": 5,    # 5 requests per minute
    "basic": 15,  # 15 requests per minute
    "premium": 60 # 60 requests per minute
}

# In-memory rate limit storage (use Redis in production)
rate_limit_store = {}

# Fake user database for demo (use a real database in production)
fake_users_db = {
    "demo": {
        "username": "demo",
        "full_name": "Demo User",
        "email": "demo@example.com",
        "hashed_password": hashlib.sha256("password".encode()).hexdigest(),
        "disabled": False,
        "tier": "free"
    },
    "premium": {
        "username": "premium",
        "full_name": "Premium User",
        "email": "premium@example.com",
        "hashed_password": hashlib.sha256("premium".encode()).hexdigest(),
        "disabled": False,
        "tier": "premium"
    }
}

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    tier: str = "free"

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

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
    resume_sections: Optional[Dict] = None  # Added to match the Streamlit app response

class BatchAnalysisRequest(BaseModel):
    job_details: JobDetails
    resume_ids: List[str]
    
class SkillRequest(BaseModel):
    name: str
    aliases: Optional[List[str]] = None
    category: Optional[str] = None
    related: Optional[List[str]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Resume Scorer API",
    description="API for analyzing resumes against job descriptions",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Auth functions
def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_user(db, username: str):
    """Get a user from the database"""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    
def authenticate_user(db, username: str, password: str):
    """Authenticate a user with username and password"""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, API_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, API_SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if the current user is active"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limit middleware based on user tier"""
    # Skip rate limiting for auth endpoints
    if request.url.path in ["/token", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    try:
        # Get auth token from header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header or not auth_header.startswith("Bearer "):
            # No token, apply strictest rate limit
            user_tier = "free"
        else:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, API_SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            user = get_user(fake_users_db, username)
            if not user:
                user_tier = "free"
            else:
                user_tier = user.tier
        
        # Get client IP for rate limiting key
        client_ip = request.client.host
        rate_key = f"{client_ip}:{user_tier}"
        
        # Check rate limit
        current_time = int(time.time())
        window_start = current_time - RATE_LIMIT_WINDOW
        
        if rate_key in rate_limit_store:
            # Clean old requests
            rate_limit_store[rate_key] = [t for t in rate_limit_store[rate_key] if t >= window_start]
            
            # Check if limit exceeded
            if len(rate_limit_store[rate_key]) >= RATE_LIMIT_MAX_REQUESTS.get(user_tier, 5):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS.get(user_tier, 5)} requests per minute."}
                )
            
            # Add current request
            rate_limit_store[rate_key].append(current_time)
        else:
            # First request
            rate_limit_store[rate_key] = [current_time]
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-Rate-Limit-Limit"] = str(RATE_LIMIT_MAX_REQUESTS.get(user_tier, 5))
        response.headers["X-Rate-Limit-Remaining"] = str(
            RATE_LIMIT_MAX_REQUESTS.get(user_tier, 5) - len(rate_limit_store.get(rate_key, []))
        )
        response.headers["X-Rate-Limit-Reset"] = str(window_start + RATE_LIMIT_WINDOW)
        return response
    except Exception as e:
        logger.error(f"Error in rate limiting: {e}")
        return await call_next(request)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get an access token for API authentication"""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the currently authenticated user"""
    return current_user

# Authenticated analyze endpoint (original - kept for backward compatibility)
@app.post("/analyze/auth", response_model=ResumeAnalysisResponse)
async def analyze_resume_auth_endpoint(
    resume: UploadFile = File(...),
    job_details: JobDetails = Depends(),
    translate: bool = Form(False),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze a resume against job details (authenticated version)
    
    - **resume**: PDF resume file
    - **job_details**: Job details including summary, duties, skills, etc.
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
        
        logger.info(f"Extraction result: {extraction_result}")
        
        if not extraction_result:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the PDF - extraction result is None"
            )
            
        if not extraction_result.get("text"):
            # Fix the case where text is None - convert to empty string
            extraction_result["text"] = ""
        
        # Convert job details to dict
        job_dict = job_details.dict(exclude_unset=True)
        
        # Ensure job details are not empty
        if not any(job_dict.values()):
            job_dict = {
                "summary": "Not provided",
                "skills": "Not specified"
            }
        
        # Analyze the resume
        try:
            analysis = analyze_resume(extraction_result, job_dict)
            
            # Get resume sections for enhanced response
            resume_sections = extract_resume_sections(extraction_result["text"])
            if resume_sections:
                analysis["resume_sections"] = resume_sections
                
            return analysis
        except Exception as e:
            # Detailed debugging for analysis errors
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Analysis error: {e}\n{error_trace}")
            
            # Return a simplified analysis with error
            return {
                "error": str(e),
                "match_percentage": "0%",
                "recommendation": "Error during analysis",
                "skills_match": {},
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {}
            }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error analyzing resume: {e}\n{error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# New analyze endpoint (public) - matches the Streamlit interface
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
        
        logger.info(f"Extraction result: {extraction_result}")
        
        if not extraction_result:
            return JSONResponse(
                status_code=422,
                content={"error": "Could not extract text from the PDF", 
                         "match_percentage": "0%",
                         "recommendation": "Please provide a valid PDF resume",
                         "skills_match": {},
                         "experience": {},
                         "education": {},
                         "certifications": {},
                         "keywords": {},
                         "industry": {}}
            )
            
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
        
        # Analyze the resume
        try:
            analysis = analyze_resume(extraction_result, job_dict)
            
            # Get resume sections for enhanced response
            resume_sections = extract_resume_sections(extraction_result["text"])
            if resume_sections:
                analysis["resume_sections"] = resume_sections
                
            return analysis
            
        except Exception as e:
            # Return a simplified analysis with error
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Analysis error: {e}\n{error_trace}")
            
            return {
                "error": str(e),
                "match_percentage": "0%",
                "recommendation": "Error during analysis",
                "skills_match": {},
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {}
            }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error analyzing resume: {e}\n{error_trace}")
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e),
                    "match_percentage": "0%",
                    "recommendation": "Server error during analysis",
                    "skills_match": {},
                    "experience": {},
                    "education": {},
                    "certifications": {},
                    "keywords": {},
                    "industry": {}}
        )

@app.post("/batch-analyze")
async def batch_analyze_endpoint(
    request: BatchAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Process multiple resume IDs in batch
    
    - **job_details**: Job details including summary, duties, skills, etc.
    - **resume_ids**: List of resume IDs to process
    
    Note: This endpoint assumes the resumes are already in the system.
    In a real implementation, you'd have a storage service to retrieve them.
    """
    # This is a placeholder. In a real implementation, you would:
    # 1. Retrieve the resumes from your storage (S3, database, etc.)
    # 2. Process them in parallel
    # 3. Return the results
    
    return {
        "status": "processing",
        "message": f"Processing {len(request.resume_ids)} resumes against job requirements",
        "job_id": "batch-" + hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    }

@app.get("/sections/{resume_id}")
async def get_resume_sections(
    resume_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Extract common sections from a resume
    
    - **resume_id**: ID of a resume already in the system
    
    Note: This endpoint assumes the resume is already in the system.
    """
    # This is a placeholder. In a real implementation, you would:
    # 1. Retrieve the resume from your storage
    # 2. Extract the sections
    # 3. Return them
    
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="This endpoint is a placeholder and not yet implemented"
    )

# Authenticated skills endpoint
@app.get("/skills/auth")
async def list_skills_auth(
    current_user: User = Depends(get_current_active_user)
):
    """
    List all skills in the ontology (authenticated)
    """
    skill_ontology = get_skill_ontology()
    return {"skills": list(skill_ontology.skills_map.keys())}

# Public skills endpoint
@app.get("/skills")
async def list_skills():
    """
    List all skills in the ontology (public version)
    """
    skill_ontology = get_skill_ontology()
    return {"skills": list(skill_ontology.skills_map.keys())}

@app.post("/skills")
async def add_skill(
    skill: SkillRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Add a new skill to the ontology
    
    - **name**: Name of the skill
    - **aliases**: Alternative names
    - **category**: Category of the skill
    - **related**: Related skills
    """
    # Check if user has permission (premium users only)
    if current_user.tier != "premium":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only premium users can add skills to the ontology"
        )
    
    skill_ontology = get_skill_ontology()
    success = skill_ontology.add_skill(
        skill.name,
        skill.aliases,
        skill.category,
        skill.related
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add skill to the ontology"
        )
    
    return {"status": "success", "message": f"Added skill: {skill.name}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check system memory usage - return degraded if high
        import psutil
        memory = psutil.virtual_memory()
        status = "ok"
        if memory.percent > 90:
            status = "degraded"
            
        # Return minimal response to save memory
        return {
            "status": status,
            "timestamp": int(time.time()),
            "memory_usage": memory.percent
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "timestamp": int(time.time()),
            "error": str(e)
        }

# Debug endpoint for PDF extraction
@app.post("/debug/extract-pdf")
async def debug_extract_pdf(
    resume: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Debug endpoint to extract text from a PDF resume
    
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
        if extraction_result and extraction_result.get("text"):
            sections = extract_resume_sections(extraction_result["text"])
        else:
            sections = {}
        
        # Clean up
        os.unlink(temp_file_path)
        
        return {
            "status": "success",
            "full_result": extraction_result,
            "sections": sections
        }
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# For running the API directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000) 