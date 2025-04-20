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
```

## Deployment

### API Deployment to Render

The Resume Scorer API is ready for Render deployment. For detailed instructions, please refer to the [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) file.

Key features of the Render deployment:
- Optimized for smaller footprint (under 250MB)
- Uses CPU-only versions of PyTorch and other large libraries
- Includes persistent storage for uploaded files and cached models
- Free tier compatible

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

The system provides a REST API that can be accessed without authentication, matching the Streamlit interface functionality:

```bash
# Analyze a resume (public endpoint)
curl -X POST "https://your-api-url.onrender.com/analyze" \
  -F "resume=@path/to/resume.pdf" \
  -F "job_summary=Job summary text" \
  -F "key_duties=Key responsibilities" \
  -F "essential_skills=Required skills" \
  -F "qualifications=Required qualifications"

# List all skills
curl -X GET "https://your-api-url.onrender.com/skills"

# Extract text and sections from a resume
curl -X POST "https://your-api-url.onrender.com/debug/extract" \
  -F "resume=@path/to/resume.pdf"

# Health check
curl -X GET "https://your-api-url.onrender.com/health"
```

For more details on the API endpoints, request/response formats, and examples, see the [API_README.md](API_README.md) file.

## Key Features & Improvements

### Performance & Scalability

-   **Upgraded Model**: Powerful `all-mpnet-base-v2` model with ONNX runtime quantization for faster inference
-   **Parallel Processing**: Multi-core resume processing with joblib for batch analysis
-   **Database-Backed Cache**: SQLite database for faster lookups of analysis results and embeddings
-   **Consolidated API**: Unified API implementation for both public and authenticated access

### Accuracy & Analysis

-   **Skill Ontology**: Normalizes skill variations (e.g., "JS" → "JavaScript") with context-aware proficiency detection
-   **Enhanced Experience Extraction**: Detects job titles and extracts employment duration with total experience calculation
-   **Education & Certification Analysis**: Validates degrees and certifications with impact assessment
-   **Semantic Matching**: Two-stage matching process with confidence metrics for extracted data

### User Experience

-   **Interactive Feedback**: Tailored resume improvement suggestions with alternative skills recommendations
-   **Advanced Visualizations**: Radar charts for skills/experience/education, comparison charts, and keyword density clouds
-   **Explainability**: Confidence scores, skill proficiency details, and salary estimations
-   **Resume Section Extraction**: Automatic extraction of resume sections for structured analysis

### Integration & Deployment

-   **Public API Endpoints**: Accessible without authentication, matching Streamlit functionality
-   **Render-Ready**: Optimized for deployment on Render's cloud platform
-   **Multi-Language Support**: Detection and translation for non-English resumes
-   **Comprehensive Error Handling**: Robust error handling and logging for production use

### Additional Capabilities

-   **ATS Compliance Checking**: Format analysis and section detection
-   **Exportable Reports**: Shareable visualizations and standardized JSON exports
-   **Testing Framework**: Unit tests and test utilities for quality assurance

## Project Structure

-   `src/` - Main source code
    -   `app.py` - Streamlit web interface
    -   `app_local.py` - Offline Streamlit interface
    -   `api.py` - Legacy FastAPI REST API with authentication
    -   `utils/` - Core functionality
        -   `analyzer.py` - Resume analysis logic
        -   `pdf_extractor.py` - PDF text extraction
        -   `skill_ontology.py` - Skill normalization and detection
        -   `visualizations.py` - Chart and visualization generation
-   `api/` - API implementation
    -   `index.py` - Main API entry point (public endpoints)
    -   `requirements.txt` - API-specific dependencies
-   `requirements.txt` - Python dependencies
-   `requirements-render.txt` - Optimized dependencies for Render
-   `render.yaml` - Render configuration
-   `render-build.sh` - Build script for Render deployment
-   `RENDER_DEPLOYMENT.md` - Detailed Render deployment instructions
-   `API_README.md` - Detailed API documentation

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

-   **Render Configuration**: Easy deployment to Render's cloud platform
-   **Optimized Dependencies**: Smaller footprint for deployment
-   **Environment Configuration**: Production-ready environment settings

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
