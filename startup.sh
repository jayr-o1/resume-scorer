#!/bin/bash
# Startup script to ensure CUDA is properly disabled

# Set environment variables to disable CUDA
export CUDA_VISIBLE_DEVICES=""
export FORCE_CPU=1
export NO_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
export DISABLE_CUDA=1

# Create a Python script to patch torch and test it's working
cat > test_cuda_disabled.py <<EOF
import os
import sys

# Verify environment variables
print("Environment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"FORCE_CPU: {os.environ.get('FORCE_CPU')}")
print(f"NO_CUDA: {os.environ.get('NO_CUDA')}")
print(f"DISABLE_CUDA: {os.environ.get('DISABLE_CUDA')}")

# Try importing torch and verifying CUDA is disabled
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available (should be False): {torch.cuda.is_available()}")
    print(f"PyTorch device count (should be 0): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print("WARNING: CUDA still available even after patching!")
        sys.exit(1)
    print("SUCCESS: PyTorch CUDA properly disabled")
except ImportError:
    print("PyTorch not installed - assuming CPU-only mode")
    sys.exit(0)
except Exception as e:
    print(f"Error testing PyTorch CUDA: {e}")
    print("Continuing with CPU-only mode")
    sys.exit(0)
EOF

# Run the test script first to verify CUDA is disabled
echo "Testing if CUDA is properly disabled..."
python test_cuda_disabled.py

# Always run the minimal API for now to ensure it works
echo "Running with minimal API implementation to ensure reliability..."
# First, check if api directory and minimal implementation exist
if [ -d "api" ] && [ -f "api/index.py" ]; then
    echo "Found minimal API implementation, using it"
    exec python -m uvicorn api.index:app --host=0.0.0.0 --port=${PORT:-8080} --workers=1
else
    # Create a minimal emergency API implementation
    echo "Creating emergency minimal API implementation"
    mkdir -p api
    cat > api/index.py <<EOF
import os
# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "emergency"}

@app.get("/")
async def root():
    return {"message": "API running in emergency minimal mode"}

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_summary: str = Form(""),
    key_duties: str = Form(""),
    essential_skills: str = Form(""),
    qualifications: str = Form("")
):
    try:
        # Very basic response
        return {
            "match_percentage": "0%",
            "recommendation": "API is running in emergency mode - only basic functionality available",
            "skills_match": {},
            "experience": {},
            "education": {},
            "certifications": {},
            "keywords": {},
            "industry": {}
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
EOF
    exec python -m uvicorn api.index:app --host=0.0.0.0 --port=${PORT:-8080} --workers=1
fi 