import os
import sys
import modal

# Force CPU-only mode for Modal deployment
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Define the Modal app
app = modal.App("resume-scorer-api")

# Create a custom image that includes all our dependencies
image = modal.Image.debian_slim().pip_install([
    "PyPDF2==3.0.1",
    "python-dotenv==1.0.0",
    "pdfplumber==0.10.3",
    "transformers==4.30.2",  # Use older version
    "torch==2.0.1",  # Use older version
    "huggingface_hub==0.15.1",  # Older version with cached_download
    "sentence-transformers==2.2.2",
    "spacy==3.5.3",
    "scikit-learn==1.2.2",
    "numpy==1.24.3",
    "tqdm>=4.65.0",
    "pandas>=2.0.0",
    "joblib==1.2.0",
    "packaging>=23.0",
    "langdetect>=1.0.9",
    "fastapi>=0.110.0",
    "uvicorn>=0.25.0",
    "python-multipart>=0.0.6",
    "pdfminer.six==20221105",
    "pymupdf==1.22.5",
    "thefuzz==0.19.0",
    "psutil==5.9.5",
    "matplotlib>=3.7.0",  # Add matplotlib for visualizations
    "altair>=4.0.0",      # Add altair for visualizations
    "wordcloud>=1.9.0",   # Add wordcloud for visualizations
    "redis>=5.0.0"        # Add redis for caching
])

# Download spaCy model
image = image.run_commands("python -m spacy download en_core_web_sm")

# Add local directory to the image
image = image.add_local_dir(".", remote_path="/app")

@app.function(image=image)
def debug():
    """Run debug information to help diagnose issues."""
    import os
    import sys
    
    os.chdir("/app")
    
    # Add /app to Python path
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    
    try:
        from debug_info import print_debug_info
        print_debug_info()
    except Exception as e:
        print(f"Error running debug info: {e}")
        print("Trying manual debug...")
        
        print("\nPython path:")
        for p in sys.path:
            print(f"  - {p}")
            
        print("\nCurrent directory and contents:")
        print(f"  - {os.getcwd()}")
        try:
            print("  - Files and directories:")
            for item in os.listdir("."):
                print(f"    - {item}")
        except Exception as e:
            print(f"  - Error listing directory: {e}")

# Define the web app to be served
@app.function(
    image=image,
    timeout=600,  # 10 minutes timeout for model loading and inference
    memory=4096,  # 4GB memory allocation
    gpu=None,     # No GPU needed
    min_containers=1,  # Keep one container warm for faster responses
)
@modal.asgi_app()
def fastapi_app():
    """Create and serve the FastAPI application."""
    import os
    import sys
    import importlib.util
    
    # Set the current directory to /app (where code is mounted)
    os.chdir("/app")
    
    # Add the /app directory to sys.path so Python can find the modules
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Try to import local API directly
    if os.path.exists("local_api") and os.path.exists("local_api/app.py"):
        print("Found local_api directory and app.py file")
        
        # Try to load the module directly
        try:
            print("Trying to import app directly...")
            spec = importlib.util.spec_from_file_location("local_api.app", "local_api/app.py")
            local_api_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_api_app)
            
            if hasattr(local_api_app, "app"):
                print("Successfully imported app from local_api.app!")
                return local_api_app.app
            else:
                print("app object not found in the local_api.app module")
        except Exception as e:
            print(f"Error importing local_api.app directly: {e}")
    else:
        print(f"local_api directory or app.py not found")
    
    # Create a fallback FastAPI app
    from fastapi import FastAPI
    fallback_app = FastAPI()
    
    @fallback_app.get("/")
    def root():
        return {
            "message": "Error loading main app. Module 'local_api.app' could not be imported.",
            "python_path": sys.path,
            "current_directory": os.getcwd(),
            "directory_contents": os.listdir("."),
            "local_api_exists": os.path.exists("local_api"),
            "local_api_contents": os.listdir("local_api") if os.path.exists("local_api") else "Directory not found"
        }
    
    return fallback_app

if __name__ == "__main__":
    print("To deploy the Resume Scorer API to Modal, run:")
    print("modal deploy modal_deploy.py")
    print("To run debug information, run:")
    print("modal run modal_deploy.py::debug") 