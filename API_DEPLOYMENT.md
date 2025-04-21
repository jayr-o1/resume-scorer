## API Deployment Guide

This guide outlines the steps to deploy the Resume Scorer API, either locally or on Render.

### Directory Structure

The API has a modular structure with different implementations:

```
resume-scorer/
├── api/                   # Legacy API implementation
├── local_api/             # Local development API
│   ├── __init__.py
│   └── app.py             # FastAPI app for local development
├── render_api/            # Render deployment API
│   ├── __init__.py
│   └── app.py             # FastAPI app optimized for Render
├── src/                   # Core source code
│   ├── api.py             # Original API implementation
│   ├── utils/             # Common utilities
│   └── ...
├── run_api.py             # Entry point script to run the API
├── render.yaml            # Render configuration
└── render-build.sh        # Build script for Render deployment
```

### Running the API Locally

To run the API locally, use the `run_api.py` script with the appropriate flags:

```bash
# Run the local API implementation (recommended for development)
python run_api.py --use-local --port=8000 --debug

# Run the original API implementation
python run_api.py --use-src --port=8000 --debug

# Run the Render API implementation locally (for testing)
python run_api.py --use-render --port=8000 --debug
```

### Deploying to Render

The API is configured for deployment on Render's free tier with memory optimizations:

1. Ensure your `render.yaml` file is correctly configured
2. Push your changes to your Git repository
3. Connect your repository to Render
4. Deploy the service

The deployment uses the `render_api/app.py` implementation, which is optimized for Render's free tier limits.

### Memory Considerations

The API is optimized to run within Render's free tier memory limits:

-   Model loading is optimized for minimal memory usage
-   Number of workers is limited to 1 on the free tier
-   Memory-intensive operations are avoided
-   Background tasks are limited

### Customizing the API

To customize the API functionality:

1. For local development, modify the `local_api/app.py` file
2. For Render deployment, modify the `render_api/app.py` file
3. For shared code, update the utilities in the `src/utils/` directory

### Troubleshooting

If you encounter issues with the API:

1. Check the logs using `render logs` command or the Render dashboard
2. Verify the memory usage isn't exceeding the free tier limits (512MB)
3. Ensure the model files are correctly downloaded during deployment
4. Check that the correct API implementation is being used

### API Versioning

When making significant changes to the API, consider:

1. Updating the API version in the FastAPI app configuration
2. Documenting changes in the OpenAPI specification
3. Ensuring backward compatibility or providing migration guides

## Prerequisites

-   A [Render account](https://render.com/)
-   Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

1. Log in to your Render account
2. Click "New" and select "Web Service"
3. Connect your Git repository
4. Configure the service with the following settings:

    - **Name**: resume-scorer-api (or your preferred name)
    - **Environment**: Python
    - **Region**: Choose the closest to your users
    - **Branch**: main (or your default branch)
    - **Build Command**: `bash render-build.sh`
    - **Start Command**: `python run_api.py --port=$PORT --host=0.0.0.0`
    - **Plan**: Free (or select a paid plan if you need more resources)

5. Click "Create Web Service"

## Environment Variables

The following environment variables are set in the `render.yaml` configuration:

-   `PYTHON_VERSION`: 3.10.0
-   `PORT`: 8000

## Accessing the API

Once deployed, your API will be available at the URL provided by Render, typically:
`https://resume-scorer-api.onrender.com`

### API Endpoints

-   Health Check: `https://resume-scorer-api.onrender.com/health`
-   Resume Analysis: `https://resume-scorer-api.onrender.com/analyze` (POST request)
-   Skills List: `https://resume-scorer-api.onrender.com/skills` (GET request)

## Testing the API

You can test the API using curl, Postman, or any API testing tool:

```bash
# Check if the API is running
curl https://resume-scorer-api.onrender.com/health

# Analyze a resume
curl -X POST \
  https://resume-scorer-api.onrender.com/analyze \
  -F "resume=@/path/to/your/resume.pdf" \
  -F "job_summary=Software Engineer position"
```

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs in Render dashboard
2. Ensure your application is compatible with Python 3.10
3. Verify that all dependencies are correctly specified in requirements-render.txt

## Limitations on Free Tier

Remember that Render's free tier has the following limitations:

-   Spins down with inactivity (may cause slow initial responses)
-   750 hours free per month
-   Shared CPU
-   512 MB RAM
-   Limited storage space
