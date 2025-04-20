# Resume Scorer API

This is the API component of the Resume Scorer system, providing the same functionality as the Streamlit web interface through a RESTful API.

## Features

-   Analyze resumes against job descriptions
-   Extract skills and provide matching scores
-   Get improvement suggestions for candidates
-   Access to the skill ontology for skill normalization
-   Public endpoints that don't require authentication

## Getting Started

### Prerequisites

-   Python 3.8 or higher
-   Required packages (install from requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

### Running the API

Use the provided `run_api.py` script to start the API server:

```
python run_api.py
```

Optional parameters:

-   `--port`: Port to run the API on (default: 8000)
-   `--host`: Host to bind to (default: 0.0.0.0)
-   `--debug`: Run in debug mode with auto-reload
-   `--log-level`: Set logging level (choices: debug, info, warning, error, critical)
-   `--workers`: Number of worker processes (default: 1)
-   `--use-src`: Use the src/api.py implementation instead of api/index.py

Example:

```
python run_api.py --port 8080 --debug --log-level debug
```

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

**Response:**

```json
{
  "match_percentage": "75",
  "recommendation": "Consider",
  "skills_match": {
    "matched_skills": ["python", "javascript", "react"],
    "missing_skills": ["node.js", "express"],
    "additional_skills": ["html", "css"],
    "match_ratio": "3/5"
  },
  "experience": { ... },
  "education": { ... },
  "certifications": { ... },
  "keywords": { ... },
  "industry": { ... },
  "improvement_suggestions": { ... },
  "resume_sections": { ... }
}
```

### Skills Endpoints

#### GET /skills

List all skills in the ontology.

**Response:**

```json
{
  "skills": ["python", "javascript", "react", "node.js", ...]
}
```

#### POST /skills

Add a new skill to the ontology.

**Parameters (JSON):**

```json
{
    "name": "typescript",
    "aliases": ["ts"],
    "category": "programming_language",
    "related": ["javascript", "angular", "react"]
}
```

**Response:**

```json
{
    "status": "success",
    "message": "Added skill: typescript"
}
```

### Batch Processing

#### POST /batch-analyze

Process multiple resumes in batch.

**Parameters (form data):**

-   `job_summary`: Job summary text (optional)
-   `key_duties`: Key duties and responsibilities (optional)
-   `essential_skills`: Essential skills required (optional)
-   `qualifications`: Required qualifications and experience (optional)
-   `industry_override`: Override for auto-detected industry (optional)
-   `resumes`: List of PDF resume files

**Response:**

```json
{
    "status": "processing",
    "message": "Processing 3 resumes against job requirements",
    "batch_id": "batch-12345abc"
}
```

### Debug Endpoints

#### POST /debug/extract

Extract text and sections from a PDF resume.

**Parameters (form data):**

-   `resume`: PDF resume file

**Response:**

```json
{
    "status": "success",
    "full_result": {
        "text": "...",
        "language": "en",
        "extraction_method": "pdfplumber"
    },
    "sections": {
        "summary": "...",
        "experience": "...",
        "education": "...",
        "skills": "..."
    }
}
```

#### GET /health

Check if the API is running.

**Response:**

```json
{
    "status": "ok",
    "version": "1.0.0",
    "timestamp": "2025-04-21T12:34:56.789012"
}
```

## Deployment

### Render Deployment

The API is configured for deployment on Render. The configuration is in the `render.yaml` file.

For deployment instructions, please refer to the [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) file.

## Architecture

The API has two implementations:

1. **api/index.py**: The default implementation, focused on public access without authentication
2. **src/api.py**: The original implementation with authentication and extended features

Both implementations provide the same core functionality but with different authentication requirements.
