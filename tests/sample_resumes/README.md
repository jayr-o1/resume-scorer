# Sample Resumes for Testing

Place PDF resumes in this directory to use with the test_api.py script.

## Usage

1. Add resume PDFs to this directory
2. Run the API test script:
    ```
    python test_api.py
    ```

Or specify a particular resume:

```
python test_api.py path/to/your/resume.pdf
```

## Testing with the API

The test script will:

1. Authenticate with the API using demo credentials
2. Load a sample job description (marketing role)
3. Upload your resume for analysis
4. Display the match score and improvement recommendations

This allows you to test the Resume Scorer API functionality without building a full frontend.
