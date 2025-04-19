# AI Resume Scorer

An advanced tool that analyzes resumes against job descriptions to determine candidate suitability using local AI models.

## Features

-   PDF resume text extraction
-   Advanced analysis against job descriptions (summary, duties, skills, qualifications)
-   Industry detection and industry-specific scoring
-   Named Entity Recognition (NER) for skill identification
-   Benchmarking against industry standards
-   Resume improvement suggestions
-   Visual charts and metrics
-   Caching for faster repeat analyses
-   Web interface with modern UI elements

## Project Structure

```
resume-scorer/
├── src/                      # Main application directory
│   ├── utils/                # Utility functions
│   │   ├── pdf_extractor.py  # PDF text extraction
│   │   └── analyzer.py       # Enhanced analysis engine
│   ├── data/                 # Sample data
│   │   ├── sample_resume.txt # Sample resume text file
│   │   ├── sample_resume.pdf # Sample resume PDF
│   │   ├── failing_resume.txt # Example of a poor match resume
│   │   ├── failing_resume.pdf # PDF of poor match resume
│   │   └── tech_job.json     # Sample tech job description
│   ├── templates/            # Any templates for outputs
│   └── app.py                # Streamlit web application
├── scripts/                  # Utility scripts
│   └── create_pdf.py         # Script to create PDF from text
├── model_cache/              # Cache directory for faster analysis
├── requirements.txt          # Dependencies
└── .gitignore                # Git ignore file
```

## Setup

1. Clone this repository
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. If spaCy models are not automatically installed, run:
    ```
    python -m spacy download en_core_web_sm
    ```

## Usage

### Web Interface

Run the Streamlit application:

```
cd src
streamlit run app.py
```

The web interface allows you to:

-   Upload a resume PDF
-   Enter job details manually or use provided samples
-   Select or auto-detect the industry
-   Get detailed analysis with visualizations
-   View improvement suggestions
-   See benchmarking against industry standards

### Create Sample PDF

Generate a PDF from a text file resume:

```
cd scripts
python create_pdf.py path/to/text_file.txt output_file.pdf
```

## How it Works

The system:

1. Extracts text from a PDF resume
2. Detects or allows manual selection of industry
3. Uses embedding models for semantic comparison
4. Employs NER for enhanced skill extraction
5. Calculates weighted scores based on industry standards
6. Benchmarks against industry expectations
7. Provides detailed analysis with visualizations
8. Generates tailored improvement suggestions

## Analysis Components

-   **Industry Detection**: Automatically identifies the most relevant industry
-   **Semantic Analysis**: Uses sentence-transformers for advanced text comparison
-   **NER Skill Detection**: Finds skills not explicitly mentioned
-   **Industry-Based Weighting**: Different score weighting per industry
-   **Benchmarking**: Compares resume against industry standards
-   **Caching**: Stores analysis results for faster repeat processing
-   **Improvement Suggestions**: Tailored recommendations to enhance the resume

## Output Format

```
AI Insights
Match Percentage: XX%
Industry: [Detected Industry]
Skills Match: X/Y skills matched
Additional Skills Detected: [Skills found via NER]
Experience: Required vs. Applicant comparison
Education: Qualification assessment
Certifications: Any relevant certifications
Resume Keywords: Keyword match rate
Industry Benchmarks: Skills, Experience, Education, Overall
Improvement Suggestions: Specific recommendations
Recommendation: Hire recommendation status
```

## Technology Stack

-   **Python**: Core programming language
-   **Streamlit**: Web interface
-   **Sentence-Transformers**: Semantic text comparison
-   **SpaCy**: Named Entity Recognition
-   **PDFPlumber/PyPDF2**: PDF text extraction
-   **Scikit-learn**: Machine learning utilities
-   **Pandas & Altair**: Data processing and visualization
