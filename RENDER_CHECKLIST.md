# Render Deployment Checklist

This checklist ensures that the Resume Scorer is properly configured for deployment on Render with all optimizations enabled.

## Pre-Deployment Checks

-   [x] Updated `render.yaml` with:

    -   [x] Resource limits (CPU, memory)
    -   [x] Scaling configuration
    -   [x] Health check path
    -   [x] Proper startup command
    -   [x] Environment variables for optimizations

-   [x] Updated `requirements-render.txt` with:

    -   [x] All required packages
    -   [x] `psutil` for memory monitoring
    -   [x] `pympler` for memory profiling
    -   [x] `bitsandbytes` and `accelerate` for model quantization

-   [x] Ensured `render-build.sh` includes:

    -   [x] Model downloading and caching
    -   [x] Setting up proper directory structure
    -   [x] Installing system dependencies

-   [x] Enhanced code with:
    -   [x] Memory optimization with dynamic threshold detection
    -   [x] Model warmup during startup
    -   [x] Improved skill matching with strict word boundaries
    -   [x] Batch processing capabilities
    -   [x] Dynamic model quantization based on available resources

## Deployment Steps

1. Push the updated code to your GitHub repository

    ```
    git add .
    git commit -m "Add deployment optimizations for Render"
    git push
    ```

2. Deploy to Render using one of these methods:

    ### A. Using the Blueprint (Recommended)

    - Go to your Render Dashboard
    - Click "New" → "Blueprint"
    - Connect your GitHub repository
    - Render will detect the `render.yaml` file and set up services automatically

    ### B. Manually Creating the Web Service

    - Go to your Render Dashboard
    - Click "New" → "Web Service"
    - Connect your GitHub repository
    - Configure with these settings:
        - **Name**: resume-scorer-api
        - **Environment**: Python
        - **Build Command**: `./render-build.sh`
        - **Start Command**: `python run_api.py --port=$PORT --host=0.0.0.0 --workers=2 --use-render --preload-models --use-quantization --task-specific-models --optimize-memory`
        - **Plan**: Free or Starter

3. Verify proper configuration in environment variables:
    - `TRANSFORMERS_OFFLINE=1`
    - `MEMORY_CONSTRAINED=1`
    - `MODEL_WARMUP_ON_START=true`
    - `ENABLE_DYNAMIC_RESOURCE_MANAGEMENT=true`
    - `USE_QUANTIZED_MODEL=1`
    - `USE_TASK_SPECIFIC_MODELS=1`
    - `OPTIMIZE_MEMORY=1`

## Post-Deployment Verification

1. Check build logs for successful:

    - Model downloading
    - Model quantization preparation
    - No errors during build process

2. Test the deployed API:

    - Health check endpoint: `https://your-app.onrender.com/health`
    - Test analysis with sample data using the included client script:
        ```
        python api_client_example.py --url https://your-app.onrender.com
        ```

3. Monitor performance:
    - Watch memory usage in Render dashboard
    - Check logs for any memory warnings
    - Verify first request latency is acceptable

## Troubleshooting

-   **Memory issues**: If you see OOM errors, consider:

    -   Increasing memory limits in `render.yaml`
    -   Adjusting worker count (reduce to 1)
    -   Enabling more aggressive memory cleanup

-   **Slow cold start**:

    -   Free tier will always have some cold start delay
    -   Ensure model warmup is working correctly
    -   Consider upgrading to a paid plan for faster cold starts

-   **Model loading failures**:
    -   Check logs for specific error messages
    -   Ensure models were properly cached during build
    -   Verify all model dependencies are installed

## Next Steps

-   Set up monitoring or alerts with Render's notification settings
-   Consider configuring autoscaling for production loads
-   Implement CloudFront or similar CDN if serving UI assets
