# Deploying Resume Scorer API to Modal

This guide explains how to deploy the Resume Scorer API to Modal, a cloud platform for running Python functions.

## Prerequisites

1. Create a Modal account at [modal.com](https://modal.com)
2. Install the Modal CLI:
    ```bash
    pip install modal
    ```
3. Authenticate with the Modal CLI:
    ```bash
    modal token new
    ```

## Deployment Steps

1. Ensure you have all required dependencies installed:

    ```bash
    pip install -r requirements.txt
    ```

2. Deploy the API to Modal:

    ```bash
    modal deploy modal_deploy.py
    ```

3. Once deployed, Modal will provide you with a URL where your API is hosted. The API will be available at endpoints like:
    - `https://<your-modal-app-url>/analyze` - Analyze a single resume
    - `https://<your-modal-app-url>/batch-analyze` - Analyze multiple resumes
    - `https://<your-modal-app-url>/health` - Health check endpoint

## Technical Details

The Modal deployment:

-   Uses a custom Docker image with all required dependencies
-   Mounts your local codebase into the container
-   Configures 4GB of memory for each container instance
-   Preloads the NLP models for faster response times
-   Keeps one container warm for immediate responses
-   Sets a timeout of 10 minutes per request to allow for model loading and analysis
-   Uses GPU acceleration if available

## Testing the API

### Using the Provided API Client

We've included a simple API client that demonstrates how to use the deployed API:

```bash
python modal_api_client.py path/to/your/resume.pdf
```

This client will:

1. Check if the API is running
2. Send the resume to the API with sample job requirements
3. Display the analysis results in a readable format

### Using cURL or Other HTTP Clients

You can also test the deployed API using curl or any HTTP client:

```bash
curl -X GET \
  https://<your-modal-app-url>/health \
  -H 'content-type: application/json'
```

For resume analysis, send a POST request with a PDF file:

```bash
curl -X POST \
  https://<your-modal-app-url>/analyze \
  -F 'resume=@/path/to/your/resume.pdf' \
  -F 'job_summary=Software Engineer with experience in Python' \
  -F 'essential_skills=Python, FastAPI, Docker'
```

## API Endpoints

The API provides the following endpoints:

1. **GET /health**

    - Check if the API is running
    - Returns: `{"status": "ok", "environment": "modal", "version": "1.0.0", "timestamp": "..."}`

2. **POST /analyze**

    - Analyze a single resume against job requirements
    - Form parameters:
        - `resume`: PDF file (required)
        - `job_summary`: Description of the job (optional)
        - `essential_skills`: Required skills (optional)
        - `qualifications`: Required qualifications (optional)
        - `industry_override`: Override auto-detected industry (optional)
        - `translate`: Whether to translate non-English resumes (optional, boolean)

3. **POST /batch-analyze**
    - Analyze multiple resumes against the same job requirements
    - Form parameters:
        - `resumes`: Multiple PDF files (required)
        - Other parameters similar to /analyze

## Monitoring and Logs

You can monitor your deployment and view logs in the Modal dashboard at [modal.com/apps](https://modal.com/apps).

## Troubleshooting

If you encounter issues:

1. Check the Modal dashboard for error logs
2. Ensure all dependencies are correctly specified in the `modal_deploy.py` file
3. Verify that your API endpoints work locally before deploying
4. Make sure your spaCy model is properly downloaded
5. Check if your Modal account has sufficient quota for your deployment

For more information, refer to the [Modal documentation](https://modal.com/docs).

## Advanced Usage

You can customize the deployment by editing the `modal_deploy.py` file:

-   Adjust memory allocation if you encounter out-of-memory errors
-   Modify the timeout for long-running requests
-   Add more dependencies if needed
-   Enable or disable GPU acceleration
