import os
import modal

# Define the Modal app
app = modal.App("resume-scorer")

# Define the image with all required dependencies
image = modal.Image.debian_slim().pip_install(
    "PyPDF2==3.0.1",
    "python-dotenv==1.0.0",
    "pdfplumber==0.10.3",
    "transformers==4.30.2",  # Downgrade to a compatible version
    "torch==2.0.1",  # Downgrade to a compatible version
    "huggingface_hub==0.15.1",  # Specific version that has cached_download
    "sentence-transformers==2.2.2",
    "spacy==3.5.3",
    "scikit-learn==1.2.2",
    "numpy==1.24.3",
    "tqdm>=4.65.0",
    "pandas>=2.0.0",
    "joblib==1.2.0",
    "packaging>=23.0",
    "langdetect>=1.0.9",
    "matplotlib>=3.7.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.25.0",
    "python-multipart>=0.0.6",
    "pdfminer.six==20221105",
    "thefuzz==0.19.0",
    "psutil==5.9.5",  # Add psutil for memory monitoring
    "pymupdf==1.22.5",  # Add pymupdf for PDF processing
    "redis>=5.0.0",  # Add redis for caching
    "wordcloud>=1.9.0",  # Add wordcloud for visualizations
    "altair>=4.0.0",  # Add altair for visualizations
)

# Download spaCy model
image = image.run_commands("python -m spacy download en_core_web_sm")

# Add local directory to the image
image = image.add_local_dir(".", remote_path="/app")

# Create a volume to store cached models and data
volume = modal.Volume.from_name("resume-scorer-volume", create_if_missing=True)
MODEL_CACHE_DIR = "/cache/model_cache"

@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: volume},
    timeout=600,
)
def download_models():
    """Download and cache models to the volume"""
    import os
    from sentence_transformers import SentenceTransformer

    # Create cache directory if it doesn't exist
    os.makedirs(f"{MODEL_CACHE_DIR}/sentence_transformers", exist_ok=True)
    
    # Download and cache the model
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=f"{MODEL_CACHE_DIR}/sentence_transformers")
    print(f"Model downloaded and cached to {MODEL_CACHE_DIR}/sentence_transformers")

# Define the FastAPI app with Modal
@app.function(
    image=image,
    volumes={MODEL_CACHE_DIR: volume},
    timeout=300,
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    # Set environment variables for the application
    os.environ["MODEL_CACHE_DIR"] = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow downloading models if needed
    os.environ["USE_TASK_SPECIFIC_MODELS"] = "1"  # Enable task-specific models
    
    # Change working directory to /app
    os.chdir("/app")
    
    # Import the FastAPI app from local_api
    import sys
    from pathlib import Path
    
    # Add the current directory to the Python path
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    
    # Import the FastAPI app
    from local_api.app import app as fastapi_app
    
    return fastapi_app

if __name__ == "__main__":
    # When running this script directly, download models to the volume
    app.run(download_models) 