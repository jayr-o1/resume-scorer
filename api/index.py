"""
Minimal API implementation that avoids CUDA dependencies
"""
import os
# Force CPU mode before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"
os.environ["SKIP_ONNX"] = "1"

import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Resume Scorer API (Minimal)", 
             description="Minimal API for analyzing resumes in PDF format",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "minimal"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Resume Scorer API is running in minimal mode"}

# Log all requests for debugging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for debugging"""
    logger.info(f"Request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_summary: str = Form(""),
    key_duties: str = Form(""),
    essential_skills: str = Form(""),
    qualifications: str = Form(""),
    industry_override: str = Form(None),
    translate: bool = Form(False)
):
    """
    Minimal analyze resume endpoint
    """
    try:
        # Log received form data
        logger.info(f"Received analyze request: file={file.filename}, job_summary_length={len(job_summary)}")
        
        # Attempt to load minimal text extraction without CUDA dependencies
        try:
            import PyPDF2
            from io import BytesIO
            
            # Basic PDF text extraction
            pdf_reader = PyPDF2.PdfReader(BytesIO(await file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                
            # Basic response
            word_count = len(text.split())
            
            # Return a response matching the expected format
            return {
                "match_percentage": "N/A",
                "recommendation": "The application is running in minimal mode due to CUDA dependency issues. Only basic text extraction is available.",
                "skills_match": {},
                "experience": {"years": "Unknown"},
                "education": {"degree": "Unknown"},
                "certifications": {},
                "keywords": {},
                "industry": {"detected": "Unknown"},
                "resume_sections": {
                    "text": text[:1000] + "..." if len(text) > 1000 else text,
                    "word_count": word_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in minimal text extraction: {e}")
            return {
                "match_percentage": "0%",
                "recommendation": "Error during analysis",
                "skills_match": {},
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {},
                "error": f"Could not perform even minimal text extraction: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error in analyze_resume: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred while processing the resume: {str(e)}"
            }
        )

# Handle form POST without specific content type
@app.post("/analyze/")
async def analyze_resume_alternative(request: Request):
    """Alternative endpoint for handling form posts with different headers"""
    try:
        # Log the request
        logger.info(f"Received alternative analyze request")
        
        # Parse form data manually
        form_data = await request.form()
        logger.info(f"Form data keys: {list(form_data.keys())}")
        
        # Get file from form data
        file = form_data.get("resume")
        if not file:
            logger.error("No file found in form data")
            return JSONResponse(
                status_code=400,
                content={"error": "No resume file provided"}
            )
        
        # Extract text using PyPDF2
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(await file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            word_count = len(text.split())
            return {
                "match_percentage": "N/A",
                "recommendation": "The application is running in minimal mode",
                "skills_match": {},
                "experience": {"years": "Unknown"},
                "education": {"degree": "Unknown"},
                "certifications": {},
                "keywords": {},
                "industry": {"detected": "Unknown"},
                "resume_sections": {
                    "text": text[:1000] + "..." if len(text) > 1000 else text,
                    "word_count": word_count
                }
            }
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return {
                "match_percentage": "0%",
                "recommendation": "Error during analysis",
                "skills_match": {},
                "experience": {},
                "education": {},
                "certifications": {},
                "keywords": {},
                "industry": {},
                "error": f"Text extraction error: {str(e)}"
            }
    except Exception as e:
        logger.error(f"Error in alternative endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle any uncaught exceptions"""
    logger.error(f"Uncaught exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"An unexpected error occurred: {str(exc)}"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting minimal API implementation")
    
    # Log Python info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Check for torch and verify CUDA is disabled
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.warning("CUDA showing as available in minimal mode! Forcing disable...")
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
    except ImportError:
        logger.info("PyTorch not available")
    except Exception as e:
        logger.warning(f"Error checking PyTorch CUDA status: {e}")
    
    # Log all available routes for debugging
    logger.info("Available routes:")
    for route in app.routes:
        logger.info(f"{route.methods} {route.path}")
    
    # Show available libraries
    try:
        import pkg_resources
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        essential_packages = ["fastapi", "uvicorn", "pydantic", "starlette", "pypdf2"]
        for pkg in essential_packages:
            if pkg in installed_packages:
                logger.info(f"Found {pkg} {installed_packages[pkg]}")
            else:
                logger.warning(f"Package {pkg} not found")
    except Exception as e:
        logger.error(f"Error checking packages: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down minimal API implementation") 