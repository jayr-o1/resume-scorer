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

### API Deployment to Vercel

The Resume Scorer API is ready for Vercel deployment with a consolidated API implementation in the `api/` directory. To deploy:

1. **Ensure you have a Vercel account** - Sign up at [vercel.com](https://vercel.com) if needed

2. **Install the Vercel CLI:**

    ```bash
    npm install -g vercel
    ```

3. **Deploy to Vercel:**

    ```bash
    vercel
    ```

4. **For production deployment:**
    ```bash
    vercel --prod
    ```

Alternatively, you can set up automatic deployments via GitHub:

1. Push your code to GitHub
2. Create a new project on Vercel and connect to your GitHub repository
3. Configure settings for your deployment
4. Deploy your project

**Important Notes for Vercel Deployment:**

-   The API uses the configuration in `vercel.json` which points to `api/index.py`
-   The API provides public endpoints that don't require authentication
-   The Vercel deployment is for the API only - the Streamlit UI must be run separately

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
curl -X POST "https://your-api-url.vercel.app/analyze" \
  -F "resume=@path/to/resume.pdf" \
  -F "job_summary=Job summary text" \
  -F "key_duties=Key responsibilities" \
  -F "essential_skills=Required skills" \
  -F "qualifications=Required qualifications"

# List all skills
curl -X GET "https://your-api-url.vercel.app/skills"

# Extract text and sections from a resume
curl -X POST "https://your-api-url.vercel.app/debug/extract" \
  -F "resume=@path/to/resume.pdf"

# Health check
curl -X GET "https://your-api-url.vercel.app/health"
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
-   **Vercel-Ready**: Optimized for serverless deployment on Vercel
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
-   `api/` - Vercel deployment API
    -   `index.py` - Main API entry point (public endpoints)
    -   `requirements.txt` - API-specific dependencies
-   `requirements.txt` - Python dependencies
-   `requirements-vercel.txt` - Optimized dependencies for Vercel
-   `vercel.json` - Vercel configuration
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

-   **Vercel Configuration**: Easy deployment to Vercel's serverless platform
-   **API Entry Point**: Clean separation of concerns for API deployment
-   **Environment Management**: Secure handling of API keys and secrets
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
