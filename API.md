# Resume Scorer API

This is the API component of the Resume Scorer capstone project, providing functionality for analyzing resumes against job descriptions.

## API Architecture

The API is implemented using FastAPI and has a simple structure:

-   `local_api/app.py`: Main API implementation with all endpoints
-   `run_api.py`: Script to run the API server with various configuration options
-   `run_local_api.py`: Simplified script to run the API for local testing
-   `test_analyzer.py`: Script to test the core analyzer functionality without the API layer

## API Endpoints

### Resume Analysis

#### POST /analyze

Analyze a resume against job requirements.

**Parameters (form data):**

-   `resume`: PDF resume file
-   `job_summary`: Job summary text (optional)
-   `key_duties`: Key duties and responsibilities (optional)
-   `essential_skills`: Essential skills required (optional)
-   `qualifications`: Required qualifications and experience (optional)
-   `industry_override`: Override for auto-detected industry (optional)
-   `translate`: Whether to translate non-English resumes to English (boolean, default: false)

### Skill Management

#### GET /skills

List all skills in the ontology.

#### POST /skills

Add a new skill to the ontology.

### Batch Processing

#### POST /batch-analyze

Process multiple resumes in batch.

### Debug Endpoints

#### POST /debug/extract

Extract text and sections from a PDF resume.

#### GET /health

Check if the API is running.

## Running the API

### Standard Method

```bash
python run_api.py --port=8000 --debug
```

Optional parameters:

-   `--port`: Port to run the API on (default: 8000)
-   `--host`: Host to bind to (default: 0.0.0.0)
-   `--debug`: Run in debug mode with auto-reload
-   `--workers`: Number of worker processes
-   `--preload-models`: Preload models before starting
-   Various other optimization flags (see `--help`)

### Quick Testing Method

```bash
python run_local_api.py
```

This runs the API on port 8000 with auto-reload enabled.

## Testing the Core Functionality

To test if the core resume analyzer is working:

```bash
python test_analyzer.py
```

This will run a simple test that extracts text from a sample resume and performs analysis.

## Client Example

Use the provided client example to test the API:

```bash
python api_client_example.py --url http://localhost:8000
```
