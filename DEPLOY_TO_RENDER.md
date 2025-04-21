# Quick-Start Guide: Deploy Resume Scorer to Render

This quick-start guide will walk you through deploying the Resume Scorer API to Render's free tier with all optimizations enabled.

## Step 1: Push Your Code to GitHub

Ensure your code is pushed to a GitHub repository that Render can access.

## Step 2: Deploy Using the Dashboard

### Option A: Manual Setup

1. Log in to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
    - **Name**: resume-scorer-api
    - **Environment**: Python
    - **Region**: Choose closest to your users
    - **Branch**: main (or your default branch)
    - **Build Command**: `bash render-build.sh`
    - **Start Command**: `python run_api.py --port=$PORT --host=0.0.0.0 --workers=1 --use-render --preload-models --use-quantization --task-specific-models --optimize-memory`
    - **Plan**: Free
5. Add the following environment variables:
    - `ENVIRONMENT`: production
    - `RENDER`: true
    - `TRANSFORMERS_OFFLINE`: 1
    - `HF_DATASETS_OFFLINE`: 1
    - `PYTHONMALLOC`: malloc
    - `MALLOC_TRIM_THRESHOLD_`: 65536
    - `PYTORCH_JIT`: 0
    - `USE_QUANTIZED_MODEL`: 1
    - `USE_TASK_SPECIFIC_MODELS`: 1
    - `OPTIMIZE_MEMORY`: 1
6. Click "Create Web Service"

### Option B: Using render.yaml (Recommended)

1. Log in to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the render.yaml file and set up your service
5. Click "Apply" to create the service

## Step 3: Monitor the Build

The build process will:

1. Install dependencies (including bitsandbytes for quantization)
2. Download and cache the required models
3. Optimize for offline use
4. Set up the API service

The first build may take 5-10 minutes, especially with model downloading.

## Step 4: Test Your API

Once the deployment is complete:

1. Navigate to the URL provided by Render (e.g., https://resume-scorer-api.onrender.com)
2. Test the health endpoint: `https://resume-scorer-api.onrender.com/health`
3. Test a basic analysis using the provided `api_client_example.py` script (update the URL)

## Optimization Details

This deployment automatically enables:

-   **Model Quantization**: 8-bit precision models (75% smaller)
-   **Task-Specific Models**: Optimized models for different analysis tasks
-   **Memory Optimizations**: Aggressive memory management
-   **Model Pre-loading**: Models downloaded during build
-   **Offline Mode**: No runtime downloads

## Troubleshooting

-   **Memory Issues**: Check logs for "OOM" (Out of Memory) errors - if present, try disabling quantization
-   **Slow Response**: First request after inactivity will be slow (free tier limitation)
-   **Build Failures**: Check build logs for specific error messages
-   **Model Loading Issues**: Look for errors related to transformers or bitsandbytes

## Monitoring

Monitor your service through the Render dashboard:

-   Check resource usage (especially memory)
-   Review logs for memory warnings
-   Monitor response times

For detailed deployment options, see [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).
