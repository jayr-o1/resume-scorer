# Resume Scorer API Deployment Guide

This guide explains how to deploy the Resume Scorer API on Render.

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
