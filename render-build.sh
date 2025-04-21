#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies
if [ -f apt.txt ]; then
  cat apt.txt | xargs apt-get update && apt-get install -y
fi

# Set environment variables for optimized build
export PYTHONHASHSEED=0
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Explicitly disable CUDA to prevent any GPU libraries from being loaded
export CUDA_VISIBLE_DEVICES=""
export FORCE_CPU=1 
export NO_CUDA=1
export DISABLE_CUDA=1

# Install pip-tools to handle dependencies better
pip install --no-cache-dir pip-tools

# Install basic dependencies first
echo "Installing basic dependencies..."
pip install --no-cache-dir wheel setuptools-rust

# Handle problematic packages separately
echo "Installing onnxruntime separately..."
pip install --no-cache-dir onnxruntime || pip install --no-cache-dir onnxruntime-cpu || {
    echo "Failed to install onnxruntime-cpu, will continue without it"
}

# Uninstall any existing CUDA-enabled PyTorch first
echo "Removing any existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio

# Install CPU-only version of PyTorch
echo "Installing CPU-only PyTorch..."
pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Setup cargo home in user directory to avoid permission issues
export CARGO_HOME="$HOME/.cargo"
mkdir -p $CARGO_HOME/registry/cache
mkdir -p $CARGO_HOME/registry/index
mkdir -p $CARGO_HOME/git

# Configure cargo to use these directories
cat > $CARGO_HOME/config.toml << EOL
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"

[registry]
index = "https://github.com/rust-lang/crates.io-index"

[source]
directory = "$CARGO_HOME"
EOL

echo "CARGO_HOME set to $CARGO_HOME"

# Check rust installation
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust for maturin compilation..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    source "$CARGO_HOME/env"
fi

# Install dependencies with optimized requirements for Render
echo "Installing dependencies with memory optimization..."
# Use requirements-render.txt first, fall back to requirements.txt
if [ -f requirements-render.txt ]; then
    # Remove problematic onnxruntime-cpu from requirements
    grep -v "onnxruntime\|torch" requirements-render.txt > requirements-render-modified.txt
    pip install --no-cache-dir -r requirements-render-modified.txt --no-deps || {
        echo "Failed to install all dependencies at once, trying one by one..."
        # Extract package names, removing version specifications
        cat requirements-render-modified.txt | grep -v '^\s*#' | grep -v '^\s*$' | sed 's/[>=<]=.*$//' > packages.txt
        while read -r package; do
            # Skip lines that look like options or URLs
            if [[ "$package" == --* ]] || [[ "$package" == http* ]]; then
                continue
            fi
            echo "Installing $package..."
            pip install --no-cache-dir "$package" || echo "Failed to install $package, continuing..."
        done < packages.txt
    }
else
    # Remove problematic onnxruntime from requirements
    grep -v "onnxruntime\|torch" requirements.txt > requirements-modified.txt
    pip install --no-cache-dir -r requirements-modified.txt --no-deps || {
        echo "Failed to install dependencies, trying essential packages only..."
        pip install --no-cache-dir PyPDF2 pdfplumber transformers sentence-transformers scikit-learn numpy spacy tqdm pandas fastapi uvicorn python-multipart flask gunicorn psutil
    }
fi

# Ensure psutil is installed (sometimes required for monitoring)
echo "Ensuring psutil is installed..."
pip install --no-cache-dir psutil==5.9.5

# Ensure CPU-specific dependencies
echo "Installing CPU-specific acceleration libraries..."
pip install --no-cache-dir accelerate==0.25.0 --no-deps
pip install --no-cache-dir bitsandbytes-cpu || pip install --no-cache-dir bitsandbytes==0.41.1

# Create special wrapper to avoid CUDA imports
echo "Creating PyTorch CUDA import wrapper..."
cat > /tmp/cuda_wrapper.py << EOL
import sys
import os

# Set environment variables that disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

# Create fake CUDA module to prevent import errors
class FakeCUDA:
    class cudart:
        def cudaGetDevice(*args, **kwargs):
            raise RuntimeError("CUDA is disabled")
    
    def is_available(*args, **kwargs):
        return False
    
    def device_count(*args, **kwargs):
        return 0

# Install the fake CUDA module
sys.modules["torch.cuda"] = FakeCUDA()

# Patch torch to disable CUDA
try:
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch._C._cuda_isDriverSufficient = lambda: False
    # Add any other CUDA functions that need to be stubbed
except ImportError:
    pass
EOL

# Install the wrapper as a startup script
mkdir -p $VIRTUAL_ENV/lib/python3.10/site-packages/sitecustomize/
cp /tmp/cuda_wrapper.py $VIRTUAL_ENV/lib/python3.10/site-packages/sitecustomize/__init__.py

# Cleanup pip cache to save space
rm -rf ~/.cache/pip

# Download spaCy model if not using URL in requirements.txt
if ! pip show en-core-web-sm > /dev/null; then
  python -m spacy download en_core_web_sm --no-deps
fi

# Download only essential NLTK data
echo "Downloading minimal NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Create necessary directories for model caching with proper permissions
mkdir -p model_cache
chmod -R 777 model_cache

# Ensure API directories exist
echo "Setting up API directories..."
mkdir -p local_api render_api
if [ ! -f render_api/__init__.py ]; then
  touch render_api/__init__.py
fi
if [ ! -f local_api/__init__.py ]; then
  touch local_api/__init__.py
fi

# Create a minimal API implementation for render_api
echo "Creating minimal render API implementation..."
cat > render_api/app.py << EOL
"""
Minimal version of render API that avoids CUDA dependencies
"""
import os
# Force CPU mode before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"
os.environ["SKIP_ONNX"] = "1"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Resume Scorer API", 
             description="API for analyzing resumes in PDF format",
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
    return {"status": "healthy", "mode": "CPU-only"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Resume Scorer API is running in CPU-only mode"}

# Import the full API implementation only if needed
try:
    # Try to import the analyzer module
    from api.index import app as full_app
    
    # Copy all routes from the full app
    for route in full_app.routes:
        if route.path not in ["/health", "/"]:
            app.routes.append(route)
            
    logger.info("Successfully imported full API implementation")
except Exception as e:
    logger.error(f"Error importing full API: {e}")
    logger.info("Running with minimal API implementation")
    
    @app.post("/analyze")
    async def analyze_fallback(file: UploadFile = File(...)):
        """Fallback analyze endpoint"""
        return {
            "error": "Full API implementation could not be loaded",
            "message": "The system is currently running in minimal mode due to initialization errors."
        }
EOL

# Print current Python packages
echo "Installed Python packages:"
pip list

# Enable offline mode for Hugging Face during model download
export TRANSFORMERS_CACHE="$PWD/model_cache"
export HF_HOME="$PWD/model_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=0  # Allow online downloads during setup

# Fix for Hugging Face connectivity issues
echo "Setting up Hugging Face models with backup downloads..."
mkdir -p model_cache/huggingface

# Try direct model download first with CUDA disabled
python -c "
import os
# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CPU'] = '1'
os.environ['NO_CUDA'] = '1'

from sentence_transformers import SentenceTransformer
import torch

# Verify we're using CPU
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch version: {torch.__version__}')

# Set model cache directory
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(os.getcwd(), 'model_cache', 'sentence_transformers')

try:
    print('Attempting to download SentenceTransformer model...')
    # Use all-MiniLM-L6-v2 as it's smaller and works well
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Successfully downloaded model')
    # Test the model
    test_embedding = model.encode(['Test sentence'])
    print(f'Model successfully tested with shape {test_embedding.shape}')
except Exception as e:
    print(f'Error downloading model: {e}')
" || echo "Warning: Failed to download sentence transformer model"

# Skip trying to run task specific and quantized model downloads if the main model failed
echo "Setting environment variables for offline use..."
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export PYTORCH_JIT=0

# Clean up any temporary files to reduce image size
echo "Cleaning up temporary files..."
rm -rf /tmp/* 2>/dev/null || true
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true

echo "Build completed with optimizations for Render free tier"
if [ -d model_cache/sentence_transformers ]; then
  echo "Model cache contains: $(ls -l model_cache/sentence_transformers)"
else
  echo "Warning: model_cache/sentence_transformers directory does not exist"
fi
echo "Total model cache size: $(du -sh model_cache 2>/dev/null || echo 'Cannot determine cache size')"

# Show file system
echo "Checking file system permissions"
ls -la /usr/local/cargo/ || true

# Update pip
pip install --upgrade pip

# Install the packages without the problematic maturin dependency
echo "Installing dependencies..."
pip install -r requirements.txt || {
    echo "Standard install failed, trying alternative approach"
    # Try installing critical dependencies one by one
    pip install torch
    pip install transformers
    pip install fastapi
    pip install uvicorn
    pip install python-multipart
    pip install nltk
    pip install scikit-learn
    
    # Install any remaining dependencies
    pip install -r requirements.txt --no-deps
}

# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords');"

# Make sure the model directories exist
mkdir -p model_cache 