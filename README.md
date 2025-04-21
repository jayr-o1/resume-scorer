# Resume Scorer

Resume Scorer is an AI-powered tool that analyzes resumes against job descriptions to help job seekers improve their applications and help employers screen candidates efficiently.

## Key Features

-   **Resume Analysis**: Perform detailed semantic analysis of resumes against job requirements
-   **Skill Matching**: Identify matched and missing skills with detailed breakdown
-   **Experience Analysis**: Evaluate work experience relevance to the position
-   **Education Assessment**: Compare educational qualifications to job requirements
-   **Improvement Suggestions**: Get actionable recommendations to improve application
-   **Industry Benchmarking**: Compare resume scores against industry standards
-   **Memory Optimized**: Runs efficiently on systems with limited resources
-   **API Interface**: Access functionality via REST API
-   **CPU-only Mode**: Works without requiring GPU acceleration

## Architecture

Resume Scorer uses a combination of LLM-based semantic analysis, named entity recognition, and domain-specific heuristics to provide accurate resume assessments:

-   **Document Processing**: Extract and structure text from PDF resumes
-   **Semantic Matching**: Compare resume content to job requirements using embeddings
-   **Skill Ontology**: Normalize and match skills across different naming conventions
-   **Optimization Layer**: Memory-efficient implementations for resource-constrained environments

These optimizations make Resume Scorer suitable for deployment on platforms with memory constraints.

## Components

The project consists of the following main components:

-   **Core Analyzer**: The underlying resume analysis engine
-   **API**: REST API for automated resume analysis
-   **Memory Monitor**: Monitors and manages memory usage during analysis

## Requirements

-   Python 3.8 or higher
-   Dependencies listed in requirements.txt
-   Minimum 1GB RAM (2GB+ recommended)
-   CPU with 2+ cores recommended
-   Storage: 500MB for base models, 1GB+ recommended

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/resume-scorer.git
cd resume-scorer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### API

Run the API server for resume analysis:

```bash
python run_api.py --port=8000 --debug
```

For quick testing in development:

```bash
python run_local_api.py
```

API documentation is available at `http://localhost:8000/docs` when the server is running.

For details on API endpoints and usage, see [API.md](API.md).

### API Client

The repository includes a sample API client for testing:

```bash
python api_client_example.py --url http://localhost:8000
```

## Testing

To verify that the analyzer is working correctly:

```bash
python test_analyzer.py
```

## Performance Optimization

Resume Scorer includes several optimizations for performance and memory usage:

-   **Quantized Models**: Optional 8-bit quantization for reduced memory usage
-   **Task-specific Models**: Use specialized models for different analysis tasks
-   **Memory Monitoring**: Active monitoring of memory usage with cleanup
-   **Cache Management**: Intelligent caching with automatic trimming
-   **CPU-only Mode**: Patched to run without GPU/CUDA dependencies

Enable these optimizations with command-line flags:

```bash
python run_api.py --optimize-memory --use-quantization --task-specific-models
```

## Project Structure

-   `src/` - Core source code
    -   `utils/` - Utility modules (extractors, analyzers, etc.)
    -   `data/` - Sample data and models
    -   `config.py` - Configuration settings
-   `scripts/` - Utility scripts for memory monitoring, model download, etc.
-   `local_api/` - FastAPI implementation for local development
-   `tests/` - Test files
-   `model_cache/` - Directory for cached models

## Key Files

-   `run_api.py` - Main API server script
-   `run_local_api.py` - Simplified API server for development
-   `test_analyzer.py` - Test script for core functionality
-   `requirements.txt` - Python dependencies
-   `api_client_example.py` - Example API client
-   `.env` - Environment variables (create from .env.example)

## Advanced Features

-   **Batch Processing**: Process multiple resumes against the same job description
-   **Custom Skill Ontology**: Extend the skill database with domain-specific terms
-   **Memory Profiling**: Monitor and optimize memory usage during processing
-   **Flexible Deployment**: Deploy as an API service or integrate with other systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.
