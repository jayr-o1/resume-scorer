# Resume Scorer

## About

Resume Scorer is an advanced AI-powered tool that analyzes resumes against job descriptions using natural language processing and semantic matching. The system helps recruiters and job seekers by providing objective assessments of how well a resume matches specific job requirements, offering visualization tools and actionable recommendations for improvement.

The platform combines AI, NLP, and data visualization to deliver accurate, consistent resume evaluations that eliminate human bias and increase efficiency in the hiring process. It supports both individual resume analysis and batch processing for recruitment teams.

## How It Works

1. **Text Extraction**: Enhanced PDF extraction with multiple fallback methods and language detection
2. **Skill Detection**: Context-aware skill detection using a custom skill ontology
3. **Semantic Matching**: Efficient semantic matching between resume and job requirements using all-MiniLM-L6-v2
4. **Visualization**: Generation of insights with interactive visualizations
5. **Recommendations**: Tailored improvement suggestions based on analysis results

## Performance Optimizations

Resume Scorer includes several optimizations to maximize performance and minimize resource usage:

1. **Model Quantization**: Optional 8-bit quantization reduces model size by up to 75% with minimal accuracy impact
2. **Task-Specific Models**: Specialized models for different tasks (skills, education, experience) improve both accuracy and efficiency
3. **Hybrid Approach**: Combines embedding models with rule-based extraction for optimal resource usage
4. **Memory Management**: Aggressive memory optimization with model offloading when not in use
5. **Pre-loading**: Models can be pre-loaded during startup for faster response times

These optimizations make Resume Scorer suitable for deployment on platforms with memory constraints, such as Render's free tier.

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

# Run with performance optimizations
python run.py --mode web-local --use-quantization --task-specific-models --optimize-memory

# Or for the API server with all optimizations
python run_api.py --use-quantization --task-specific-models --preload-models
```

## Testing the Application

You can quickly test the core functionality using the provided test script:

```bash
python test_analyzer.py
```

This will analyze a sample resume against a software engineering job description and show the match percentage, skills match, experience assessment, and other key metrics. The full analysis results will be saved to `analysis_results.json` for detailed inspection.

## Deployment

### API Deployment to Render

The Resume Scorer API is ready for Render deployment with memory optimizations. For detailed instructions, please refer to the [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) file.

Key optimizations for Render deployment:

-   **Model Quantization**: 8-bit quantized models reduce memory usage by up to 75%
-   **Task-Specific Models**: Different sized models for different tasks
-   **Rule-Based Components**: Non-ML approaches for suitable tasks (education/certification extraction)
-   **Offline Mode**: Pre-downloaded models prevent runtime downloads
-   **Memory Optimizations**: Advanced memory management for constrained environments

For deployment commands, see [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).

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

The system provides a REST API that can be accessed with or without authentication:

```bash
# Analyze a resume (public endpoint)
curl -X POST "http://localhost:8000/analyze" \
  -F "resume=@path/to/resume.pdf" \
  -F "job_summary=Job summary text" \
  -F "key_duties=Key responsibilities" \
  -F "essential_skills=Required skills" \
  -F "qualifications=Required qualifications"

# List all skills
curl -X GET "http://localhost:8000/skills"

# Health check
curl -X GET "http://localhost:8000/health"

# Authenticated endpoints (require token)
curl -X POST "http://localhost:8000/token" \
  -d "username=demo&password=password" \
  -H "Content-Type: application/x-www-form-urlencoded"

# Use the token for authenticated endpoints
curl -X POST "http://localhost:8000/analyze/auth" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "resume=@path/to/resume.pdf" \
  -F "job_summary=Job summary text"
```

For more details on the API endpoints, request/response formats, and examples, see the [API_README.md](API_README.md) file.

## Key Features & Improvements

### Performance & Scalability

-   **Efficient Embedding Model**: Lightweight `all-MiniLM-L6-v2` model provides excellent performance with lower resource requirements
-   **Quantization Support**: Optional 8-bit quantization for up to 75% memory reduction
-   **Task-Specific Models**: Specialized models for different analysis tasks
-   **Parallel Processing**: Multi-core resume processing with joblib for batch analysis
-   **Database-Backed Cache**: SQLite database for faster lookups of analysis results and embeddings
-   **Unified API**: Comprehensive API implementation for both public and authenticated access

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
    -   `api.py` - FastAPI REST API implementation
    -   `data/` - Sample data and skill ontology storage
    -   `utils/` - Core functionality
        -   `analyzer.py` - Resume analysis logic
        -   `pdf_extractor.py` - PDF text extraction
        -   `skill_ontology.py` - Skill normalization and detection
        -   `visualizations.py` - Chart and visualization generation
-   `api/` - API implementation for deployment
    -   `index.py` - Main API entry point (public endpoints)
    -   `requirements.txt` - API-specific dependencies
-   `tests/` - Test utilities and unit tests
-   `model_cache/` - Local cache for downloaded models
-   `scripts/` - Utility scripts
-   `requirements.txt` - Python dependencies
-   `requirements-render.txt` - Optimized dependencies for Render
-   `render.yaml` - Render configuration
-   `render-build.sh` - Build script for Render deployment
-   `run.py` - Main entry point for running the application
-   `run_api.py` - API server entry point
-   `test_analyzer.py` - Testing script for the analyzer

## Technical Implementation Details

### Model Optimization

-   **Lightweight Embedding Model**: all-MiniLM-L6-v2 provides excellent performance with significantly lower resource usage compared to larger models
-   **Embedding Caching**: Database system to cache embeddings for reuse across analyses
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
# Quick test of the analyzer
python test_analyzer.py

# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## Performance Results

Testing with the sample resume against a software engineering job description yielded the following results:

-   **Match Percentage**: 76%
-   **Recommendation**: Hire
-   **Skills Match**: Matched 2/4 required skills (JavaScript, React)
-   **Missing Skills**: Python, API Development
-   **Experience**: Required 3 years | Applicant 6 years
-   **Education**: Bachelor's degree (Meets Requirement)
-   **Processing Time**: ~5 seconds for full analysis

The all-MiniLM-L6-v2 model provides comparable results to larger models like MPNet while being much faster and using fewer resources, making it ideal for this application.

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
