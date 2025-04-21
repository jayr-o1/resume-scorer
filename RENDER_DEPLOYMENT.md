# Deploying Resume Scorer on Render

This guide will help you deploy the Resume Scorer application on Render with optimized settings for memory efficiency.

## Prerequisites

-   A [Render account](https://render.com/)
-   Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

1. Log in to your Render account
2. Click "New" and select "Web Service"
3. Connect your Git repository
4. Configure the service with the following settings:

    - **Name**: resume-scorer (or your preferred name)
    - **Environment**: Python
    - **Region**: Choose the closest to your users
    - **Branch**: main (or your default branch)
    - **Build Command**: `bash render-build.sh`
    - **Start Command**: `python run_api.py --use-quantization --task-specific-models --optimize-memory --preload-models`
    - **Plan**: Free (or select a paid plan if you need more resources)

5. Click "Create Web Service"

## Environment Variables

The following environment variables are set in the `render.yaml` configuration:

-   `PYTHON_VERSION`: 3.10.0
-   `PORT`: 8080
-   `ENVIRONMENT`: production
-   `RENDER`: true
-   `TRANSFORMERS_OFFLINE`: 1
-   `HF_DATASETS_OFFLINE`: 1
-   `PYTHONMALLOC`: malloc
-   `MALLOC_TRIM_THRESHOLD_`: 65536
-   `PYTORCH_JIT`: 0

If your application requires any additional environment variables (API keys, database credentials, etc.), add them in the Render dashboard under the "Environment" section.

## Important Notes

1. The free tier of Render has the following limitations:

    - Spins down with inactivity
    - 750 hours free per month
    - Shared CPU
    - 512 MB RAM
    - 0.5 GB persistent disk

2. If your application requires more resources, consider upgrading to a paid plan.

3. Large machine learning models may need to be downloaded during build time, which could increase build times.

## Optimizations for Render

To stay within Render's free tier limits (especially the 512MB RAM constraint), the following optimizations have been implemented:

1. **Model Quantization**: The `--use-quantization` flag enables 8-bit quantization, reducing model size by up to 75% with minimal accuracy impact
2. **Task-Specific Models**: The `--task-specific-models` flag enables specialized smaller models for different tasks:
    - General matching: all-MiniLM-L6-v2
    - Skills extraction: paraphrase-MiniLM-L3-v2 (only 17MB)
    - Education extraction: Rule-based (no ML model)
3. **Memory Optimizations**: The `--optimize-memory` flag enables several RAM-saving techniques:
    - Aggressive garbage collection
    - Memory trimming
    - System allocator usage
    - JIT compilation disabled
4. **Pre-loading**: The `--preload-models` flag ensures models are downloaded and cached during startup rather than at runtime

5. **CPU-only PyTorch**: We use the CPU-only version of PyTorch to significantly reduce the size of dependencies.

6. **Optimized Requirements**: A separate `requirements-render.txt` file is used for deployment which includes optimized versions of the dependencies.

7. **Model Offloading**: Models are unloaded when not in use to free up memory.

## Advanced Deployment with render.yaml

You can use the provided `render.yaml` configuration file for more advanced deployment settings. This file configures:

-   A web service for the API
-   A disk mount for model caching and uploaded files
-   Optimized environment variables
-   Health check endpoints

To deploy using render.yaml:

1. Push your code to a Git repository
2. Log in to Render
3. Navigate to Blueprint section
4. Click "New Blueprint Instance"
5. Connect your repository
6. Render will automatically detect the render.yaml file and configure your services

## Additional Configuration

If your application requires storage for uploaded files or model caches, you can use the `/data` directory which is mounted as a persistent disk.

## Monitoring and Debugging

The application includes built-in memory monitoring that will log when memory usage gets high and take action to reduce it. You can check these logs in the Render dashboard.

If you need to disable any optimizations for debugging:

1. Log in to Render dashboard
2. Select your web service
3. Go to Environment section
4. Modify the start command (e.g., remove specific optimization flags)

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs in Render dashboard
2. Ensure all required packages are installed in the build script
3. Verify the disk mount is configured correctly
4. Check memory usage in the logs for potential out-of-memory issues
5. If you see "Application Error" or crashes, try disabling quantization temporarily to debug
