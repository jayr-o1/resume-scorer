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
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Install dependencies with optimized requirements for Render
echo "Installing dependencies with memory optimization..."
pip install --no-cache-dir -r requirements-render.txt

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

# Pre-download and cache all models for faster startup
echo "Pre-downloading and caching models..."
python scripts/download_models.py

# Verify the models are downloaded
echo "Verifying model cache directory contents:"
find model_cache -type f | wc -l
du -sh model_cache

# Make model cache files accessible
chmod -R 755 model_cache

# Clean up any temporary files to reduce image size
echo "Cleaning up temporary files..."
rm -rf /tmp/* 2>/dev/null || true
apt-get clean
rm -rf /var/lib/apt/lists/*

# Set memory optimization variables for Python
echo "Setting memory optimization environment variables..."
export MALLOC_TRIM_THRESHOLD_=65536
export PYTHONMALLOC=malloc
export PYTORCH_JIT=0

echo "Build completed with optimizations for Render free tier" 