# AI Resume Scorer

An AI-powered tool that analyzes resumes against job descriptions to determine candidate suitability.

## Features

-   PDF resume text extraction
-   Comparison against job descriptions (summary, duties, skills, qualifications)
-   AI-powered scoring and analysis
-   Detailed match reports with percentages and recommendations
-   Web interface for easy usage

## Project Structure

```
resume-scorer/
├── src/                      # Main application directory
│   ├── utils/                # Utility functions
│   │   ├── pdf_extractor.py  # PDF text extraction
│   │   └── analyzer.py       # Resume analysis functions
│   ├── data/                 # Sample data
│   │   ├── sample_resume.txt # Sample resume text file
│   │   ├── sample_resume.pdf # Sample resume PDF
│   │   └── sample_job.json   # Sample job description
│   ├── templates/            # Any templates for outputs
│   ├── app.py                # Streamlit web application
│   └── cli.py                # Command-line interface
├── scripts/                  # Utility scripts
│   └── create_pdf.py         # Script to create PDF from text
├── requirements.txt          # Dependencies
└── .env                      # Environment variables (API keys)
```

## Setup

1. Clone this repository
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Create a `.env` file with your OpenAI API key:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```

## Usage

### Web Interface

Run the Streamlit application:

```
cd src
streamlit run app.py
```

### Command Line

Use the command-line interface for batch processing:

```
cd src
python cli.py path/to/resume.pdf --job_file path/to/job.json
```

Or specify job details directly:

```
python cli.py path/to/resume.pdf --summary "Job description" --skills "Required skills" --output results.txt
```

### Create Sample PDF

Generate a sample resume PDF from the text file:

```
cd scripts
python create_pdf.py
```

## How it Works

The system:

1. Extracts text from a PDF resume
2. Compares it against provided job details
3. Uses AI to analyze the match quality
4. Provides a detailed scoring report with recommendations

## Output Format

```
AI Insights
Match Percentage: XX%
Skills Match: X/Y skills matched
Experience: Required vs. Applicant comparison
Education: Qualification assessment
Certifications: Any relevant certifications
Resume Keywords: Keyword match rate
Recommendation: Hire recommendation status
```
