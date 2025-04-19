#!/bin/bash

# Resume Scorer Runner Script
# Usage: ./run.sh [mode] [options]
# Modes:
#   web      - Run the Streamlit web interface
#   api      - Run the FastAPI server
#   batch    - Run in batch processing mode
#   docker   - Run in Docker container

set -e

# Default configuration
PORT_WEB=8501
PORT_API=8000
MODE="web"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    web|api|batch|docker)
      MODE="$1"
      shift
      ;;
    --port)
      if [[ $MODE == "web" ]]; then
        PORT_WEB="$2"
      else
        PORT_API="$2"
      fi
      shift 2
      ;;
    --resume)
      RESUME_PATH="$2"
      shift 2
      ;;
    --job)
      JOB_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Resume Scorer Runner Script"
      echo "Usage: ./run.sh [mode] [options]"
      echo ""
      echo "Modes:"
      echo "  web      - Run the Streamlit web interface (default)"
      echo "  api      - Run the FastAPI server"
      echo "  batch    - Run in batch processing mode"
      echo "  docker   - Run in Docker container"
      echo ""
      echo "Options:"
      echo "  --port PORT    - Specify port (default: 8501 for web, 8000 for API)"
      echo "  --resume PATH  - Path to resume file (batch mode)"
      echo "  --job PATH     - Path to job details file (batch mode)"
      echo "  --help         - Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found."
    exit 1
fi

# Check for virtual environment
if [[ -d "venv" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

# Check if requirements are installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo "Installing SpaCy model..."
    python -m spacy download en_core_web_sm
fi

# Create necessary directories
mkdir -p model_cache src/data

# Run the application in the specified mode
case $MODE in
  web)
    echo "Starting Streamlit web interface on port $PORT_WEB..."
    cd src
    streamlit run app.py --server.port $PORT_WEB
    ;;
  api)
    echo "Starting FastAPI server on port $PORT_API..."
    cd src
    uvicorn api:app --host 0.0.0.0 --port $PORT_API --reload
    ;;
  batch)
    echo "Running in batch processing mode..."
    if [[ -z "$RESUME_PATH" || -z "$JOB_PATH" ]]; then
      echo "Error: --resume and --job options are required for batch mode"
      exit 1
    fi
    cd src
    python3 run.py --mode cli-local --resume "$RESUME_PATH" --job_file "$JOB_PATH"
    ;;
  docker)
    echo "Building and running in Docker container..."
    docker build -t resume-scorer .
    
    if [[ "$MODE" == "web" ]]; then
      echo "Starting Streamlit web interface in Docker on port $PORT_WEB..."
      docker run -p $PORT_WEB:8501 resume-scorer streamlit
    else
      echo "Starting FastAPI server in Docker on port $PORT_API..."
      docker run -p $PORT_API:8000 resume-scorer api
    fi
    ;;
esac 