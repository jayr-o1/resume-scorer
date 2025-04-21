#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies
if [ -f apt.txt ]; then
  cat apt.txt | xargs apt-get update && apt-get install -y
fi

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

# Set environment variables for optimized build
export PYTHONHASHSEED=0
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Install pip-tools to handle dependencies better
pip install --no-cache-dir pip-tools

# Handle maturin separately without using pip to avoid the Rust build issues
echo "Handling problematic packages..."
pip install --no-cache-dir --no-build-isolation wheel setuptools-rust

# Install dependencies with optimized requirements for Render
echo "Installing dependencies with memory optimization..."
# Use requirements-render.txt first, fall back to requirements.txt
if [ -f requirements-render.txt ]; then
    pip install --no-cache-dir -r requirements-render.txt --no-deps || {
        echo "Failed to install all dependencies at once, trying one by one..."
        # Extract package names, removing version specifications
        cat requirements-render.txt | grep -v '^\s*#' | grep -v '^\s*$' | sed 's/[>=<]=.*$//' > packages.txt
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
    pip install --no-cache-dir -r requirements.txt --no-deps || {
        echo "Failed to install dependencies, trying essential packages only..."
        pip install --no-cache-dir PyPDF2 pdfplumber transformers sentence-transformers torch scikit-learn numpy spacy tqdm pandas fastapi uvicorn python-multipart flask gunicorn psutil
    }
fi

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

# Check if the prepare_model_for_render.py script exists
if [ -f scripts/prepare_model_for_render.py ]; then
  python scripts/prepare_model_for_render.py || {
    echo "Prepare script failed, falling back to standard download"
    # Use the fallback download script if the special preparation fails
    if [ -f scripts/download_models.py ]; then
      python scripts/download_models.py
    else
      echo "No download_models.py script found. Checking sentence-transformers installation."
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
    fi
  }
else
  echo "No prepare_model_for_render.py script found, checking for download_models.py"
  if [ -f scripts/download_models.py ]; then
    python scripts/download_models.py
  else
    echo "No download_models.py script found. Using direct model initialization."
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  fi
fi

# Try to download task-specific models if the script exists
if [ -f scripts/download_models.py ]; then
  echo "Downloading task-specific models..."
  python -c "
  import os
  import sys
  sys.path.insert(0, '.') 
  os.environ['USE_TASK_SPECIFIC_MODELS'] = '1'
  try:
    from scripts.download_models import download_task_specific_models
    download_task_specific_models()
  except Exception as e:
    print(f'Error downloading task-specific models: {e}')
  "

  echo "Preparing models for quantization..."
  python -c "
  import os
  import sys
  sys.path.insert(0, '.')
  os.environ['USE_QUANTIZED_MODEL'] = '1'
  try:
    from scripts.download_models import prepare_quantized_models
    prepare_quantized_models()
  except Exception as e:
    print(f'Error preparing quantized models: {e}')
  "
else
  echo "No download_models.py script found for task-specific models."
fi

# Verify the models are downloaded
echo "Verifying model cache directory contents:"
find model_cache -type f | wc -l
du -sh model_cache
ls -la model_cache || echo "No model_cache directory found"

# Make model cache files accessible
chmod -R 755 model_cache || echo "Could not set permissions on model_cache"

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