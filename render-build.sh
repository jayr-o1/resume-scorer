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

# Install dependencies with optimized requirements for Render
echo "Installing dependencies with memory optimization..."
pip install --no-cache-dir -r requirements-render.txt

# Ensure psutil is installed (sometimes required for monitoring)
echo "Ensuring psutil is installed..."
pip install --no-cache-dir psutil==5.9.5

# Ensure bitsandbytes and accelerate for quantization
echo "Installing quantization dependencies..."
pip install --no-cache-dir bitsandbytes==0.41.1 accelerate==0.25.0

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

# Print current Python packages
echo "Installed Python packages:"
pip list

# Enable optimization flags for model download
export USE_QUANTIZED_MODEL=1
export USE_TASK_SPECIFIC_MODELS=1

# Prepare the sentence transformer model for offline use
echo "Preparing sentence transformer model for offline use..."
python scripts/prepare_model_for_render.py

# Use the fallback download script if the special preparation fails
if [ $? -ne 0 ]; then
  echo "Falling back to standard model download..."
  python scripts/download_models.py
fi

# Download task-specific models
echo "Downloading task-specific models..."
python -c "
import os
import sys
sys.path.insert(0, '.') 
os.environ['USE_TASK_SPECIFIC_MODELS'] = '1'
from scripts.download_models import download_task_specific_models
download_task_specific_models()
"

# Prepare models for quantization
echo "Preparing models for quantization..."
python -c "
import os
import sys
sys.path.insert(0, '.')
os.environ['USE_QUANTIZED_MODEL'] = '1'
from scripts.download_models import prepare_quantized_models
prepare_quantized_models()
"

# Verify the models are downloaded
echo "Verifying model cache directory contents:"
find model_cache -type f | wc -l
du -sh model_cache
ls -la model_cache/sentence_transformers/

# Make model cache files accessible
chmod -R 755 model_cache

# Set environment variables for offline use
echo "Setting environment variables for offline use..."
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export PYTORCH_JIT=0

# Clean up any temporary files to reduce image size
echo "Cleaning up temporary files..."
rm -rf /tmp/* 2>/dev/null || true
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Build completed with optimizations for Render free tier"
echo "Model cache contains: $(ls -l model_cache/sentence_transformers)"
echo "Total model cache size: $(du -sh model_cache)" 