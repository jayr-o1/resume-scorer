# Resume Scorer Optimizations for Render Deployment

This document outlines the optimizations made to ensure the Resume Scorer application runs efficiently on Render's free tier, which has memory constraints.

## Pre-downloading Models

We've implemented a model pre-downloading system to ensure models are cached during the build phase rather than at runtime:

1. **Created `scripts/download_models.py`**:

    - Downloads and caches the Sentence Transformer model (all-MiniLM-L6-v2)
    - Downloads and caches the SpaCy model (en_core_web_sm)
    - Initializes the SQLite database for caching

2. **Modified `render-build.sh`**:
    - Calls the download script during the build phase
    - Ensures proper permissions for the model cache directory
    - Verifies that models are correctly cached
    - Cleans up temporary files and pip cache
    - Sets optimization environment variables

## Memory Optimization

### Configuration Settings

Enhanced `src/config.py` file with memory optimization settings:

-   Further reduced batch sizes for text processing (now 4 instead of 8)
-   Decreased maximum text length (now 50,000 characters)
-   Reduced cache size for memory efficiency (2,000 embeddings)
-   Enabled model offloading when not in use
-   Set worker count to 1 for Render's free tier
-   Configured environment-specific settings for Render
-   Added more frequent memory monitoring on Render (every 60 seconds)

### Memory Monitoring

Enhanced `scripts/memory_monitor.py` to:

-   Track system and process memory usage more aggressively
-   Lower memory thresholds for early intervention (70%/80%/90%)
-   Perform advanced cleanup through module cache clearing
-   Use progressive cleanup strategies (standard → aggressive → emergency)
-   Identify and remove large objects when memory usage is high
-   Added malloc_trim support for better memory reclamation
-   Reduced logging frequency to save disk I/O

### Dependency Optimizations

Updated `requirements-render.txt` with memory-efficient dependencies:

-   Pinned specific versions to avoid unexpected updates
-   Used CPU-only versions of PyTorch and other ML libraries
-   Limited unnecessary dependencies
-   Used lightweight alternatives where possible
-   Added tokenizers version constraint to reduce memory usage

## Render-Specific Configurations

Updated `render.yaml` with optimized settings:

-   Set worker count to 1 to minimize memory usage
-   Enabled model preloading with the `--preload-models` flag
-   Added environment variables for offline model usage
-   Configured Python memory management variables
-   Set the disk mount point to directly map to the model cache directory
-   Added a health check endpoint for monitoring
-   Disabled auto-deployment to save resources
-   Added scaling parameters to control resources
-   Added OMP_NUM_THREADS and MKL_NUM_THREADS constraints

## Environment Variables

Added environment variables to optimize memory usage:

-   `TRANSFORMERS_OFFLINE=1` - Prevents downloading models at runtime
-   `HF_DATASETS_OFFLINE=1` - Prevents downloading datasets at runtime
-   `PYTHONMALLOC=malloc` - Uses the system allocator instead of Python's
-   `MALLOC_TRIM_THRESHOLD_=65536` - Encourages memory to be returned to the system
-   `PYTORCH_JIT=0` - Disables JIT compilation to save memory
-   `OMP_NUM_THREADS=1` - Limits OpenMP threads
-   `MKL_NUM_THREADS=1` - Limits MKL threads
-   `PYTHONGC=threshold-aggressive` - More aggressive garbage collection

## Health Monitoring

Added a memory-aware health check endpoint that:

-   Monitors system memory usage
-   Returns "degraded" status when memory is high (>90%)
-   Uses minimal response payload to save memory
-   Provides Render with proper health status information

## Testing the Optimizations

You can test these optimizations locally by:

1. Running the memory monitor:

    ```bash
    python scripts/memory_monitor.py --one_time
    ```

2. Pre-downloading models:

    ```bash
    python scripts/download_models.py
    ```

3. Running the API with optimized settings:
    ```bash
    python run_api.py --workers=1 --preload-models
    ```

## Benefits

These optimizations provide several benefits:

1. **Faster Startup**: Pre-downloaded models mean no downloading at runtime
2. **Lower Memory Usage**: Optimized dependencies and configuration reduce the memory footprint
3. **Stability**: Memory monitoring prevents crashes due to out-of-memory errors
4. **Better Performance**: Appropriate worker counts and batch sizes for the Render environment
5. **Early Warning**: Health checks identify issues before service degradation

## Monitoring and Maintenance

To monitor the application on Render:

1. Check the logs for memory usage statistics
2. Look for cleanup events in the logs, which indicate high memory usage
3. Monitor the disk usage to ensure the model cache isn't growing too large
4. Check the health endpoint status for service health

If memory issues persist, consider:

1. Further reducing the batch size in `config.py`
2. Using an even more lightweight embedding model
3. Implementing more aggressive caching strategies
4. Upgrading to a higher tier Render plan for more resources
