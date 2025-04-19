# Resume Scorer

## About

Resume Scorer is an advanced AI-powered tool that analyzes resumes against job descriptions using natural language processing and semantic matching. The system helps recruiters and job seekers by providing objective assessments of how well a resume matches specific job requirements, offering visualization tools and actionable recommendations for improvement.

The platform combines AI, NLP, and data visualization to deliver accurate, consistent resume evaluations that eliminate human bias and increase efficiency in the hiring process. It supports both individual resume analysis and batch processing for recruitment teams.

## How It Works

1. **Text Extraction**: Enhanced PDF extraction with multiple fallback methods and language detection
2. **Skill Detection**: Context-aware skill detection using a custom skill ontology
3. **Semantic Matching**: Advanced semantic matching between resume and job requirements
4. **Visualization**: Generation of insights with interactive visualizations
5. **Recommendations**: Tailored improvement suggestions based on analysis results

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-scorer.git
cd resume-scorer

# Build the Docker image
docker build -t resume-scorer .

# Run the Streamlit app
docker run -p 8501:8501 resume-scorer streamlit

# Or run the API server
docker run -p 8000:8000 resume-scorer api
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-scorer.git
cd resume-scorer

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Run the Streamlit app
cd src
streamlit run app.py

# Or run the API server
cd src
uvicorn api:app --reload
```

## Usage

### Streamlit Web Interface

The Streamlit interface provides three main tabs:

1. **Single Resume Analysis**:

    - Upload a resume PDF
    - Enter job details (summary, duties, skills, qualifications)
    - Get a comprehensive analysis with visualizations

2. **Batch Processing**:

    - Upload multiple resumes
    - Enter common job details
    - Compare and rank candidates

3. **Skill Ontology Management**:
    - View the skill ontology database
    - Add new skills with aliases and related skills

### REST API

The system also provides a REST API for integration with other systems:

```bash
# Authentication (get token)
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=password"

# Analyze a resume
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "resume=@path/to/resume.pdf" \
  -F "job_details.summary=Job summary text" \
  -F "job_details.skills=Required skills"
```

## Key Features & Improvements

### Performance & Scalability

-   **Upgraded Model**: Powerful `all-mpnet-base-v2` model with ONNX runtime quantization for faster inference
-   **Parallel Processing**: Multi-core resume processing with joblib for batch analysis
-   **Database-Backed Cache**: SQLite database for faster lookups of analysis results and embeddings

### Accuracy & Analysis

-   **Skill Ontology**: Normalizes skill variations (e.g., "JS" → "JavaScript") with context-aware proficiency detection
-   **Enhanced Experience Extraction**: Detects job titles and extracts employment duration with total experience calculation
-   **Education & Certification Analysis**: Validates degrees and certifications with impact assessment
-   **Semantic Matching**: Two-stage matching process with confidence metrics for extracted data

### User Experience

-   **Interactive Feedback**: Tailored resume improvement suggestions with alternative skills recommendations
-   **Advanced Visualizations**: Radar charts for skills/experience/education, comparison charts, and keyword density clouds
-   **Explainability**: Confidence scores, skill proficiency details, and salary estimations

### Integration & Deployment

-   **Comprehensive REST API**: FastAPI with JWT authentication, user roles, and rate limiting
-   **Multi-Language Support**: Detection and translation for non-English resumes
-   **Flexible Deployment**: Docker containerization with support for AWS, GCP, and serverless options

### Additional Capabilities

-   **ATS Compliance Checking**: Format analysis and section detection
-   **Exportable Reports**: Shareable visualizations and standardized JSON exports
-   **Testing Framework**: Unit tests and test utilities for quality assurance

## Project Structure

-   `src/` - Main source code
    -   `app.py` - Streamlit web interface
    -   `api.py` - FastAPI REST API
    -   `utils/` - Core functionality
        -   `analyzer.py` - Resume analysis logic
        -   `pdf_extractor.py` - PDF text extraction
        -   `skill_ontology.py` - Skill normalization and detection
        -   `visualizations.py` - Chart and visualization generation
-   `model_cache/` - Cached models and embeddings
-   `requirements.txt` - Python dependencies
-   `Dockerfile` - Docker containerization

## Technical Implementation Details

### Model Optimization

-   **Embedding Caching**: Database system to cache embeddings for reuse across analyses
-   **Structured Storage**: Separate tables for analysis results and embeddings
-   **Thread-safety**: Thread-local database connections for concurrent access

### Skill Ontology Design

-   **Skill Relationships**: Mapped relationships between skills to suggest alternatives for missing skills
-   **Context-aware Detection**: Detection of skill proficiency levels based on context

### Semantic Similarity Technology

-   **Two-stage Matching**: Direct pattern matching combined with semantic similarity
-   **Confidence Metrics**: Detailed confidence scoring for all extracted data points

### Deployment Infrastructure

-   **Run Scripts**: Easy running in different modes (API, UI)
-   **Deploy Scripts**: Streamlined deployment scripts for cloud environments
-   **Code Quality**: Modular architecture with clean interfaces between components

## Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   [Sentence-Transformers](https://www.sbert.net/) for embedding models
-   [SpaCy](https://spacy.io/) for NER and linguistic processing
-   [Streamlit](https://streamlit.io/) for the web interface
-   [FastAPI](https://fastapi.tiangolo.com/) for the API framework
