# Resume Scorer: Free Tier Render Optimization Summary

This document summarizes the optimizations made to ensure the Resume Scorer application runs efficiently on Render's free tier.

## Key Optimizations

### Memory Usage

1. **Reduced Batch Size**: Decreased from 8 to 4 to lower memory footprint
2. **Limited Workers**: Set to 1 worker for optimal memory usage
3. **Decreased Cache Size**: Reduced embedding cache size from 5000 to 2000
4. **Text Processing Limits**: Reduced maximum text length from 100,000 to 50,000 characters
5. **Aggressive Memory Monitoring**: Added more frequent checks (every 60s vs 300s)
6. **Memory Thresholds**: Lowered to 70%/80%/90% for earlier intervention
7. **Advanced Memory Cleanup**: Added module cache clearing and large object detection

### Performance

1. **Build Optimizations**: Clean temporary files and pip cache during build
2. **Disabled JIT**: Turned off PyTorch JIT compilation to save memory
3. **Thread Limits**: Set OMP and MKL threads to 1
4. **Aggressive GC**: Set PYTHONGC to threshold-aggressive for better reclamation

### Monitoring

1. **Memory-Aware Health Checks**: Reports "degraded" when memory is high
2. **Minimal Response Payloads**: Keep API responses small to save memory
3. **Reduced Logging**: Log only every 5th check to reduce I/O

### Reliability

1. **Disabled Auto-Deploy**: Prevents unexpected restarts
2. **Scaling Controls**: Set appropriate resource limits
3. **Memory Reclamation**: Added malloc_trim support
4. **Early Warning System**: Health checks identify issues before failures

## Testing Locally

```bash
# Test memory usage
python scripts/memory_monitor.py --one_time

# Pre-download models
python scripts/download_models.py

# Run with optimized settings
python run_api.py --workers=1 --preload-models
```

## Monitoring in Production

1. Check logs for memory usage trends
2. Monitor health endpoint status
3. Look for cleanup events in logs
4. Set up alerts for "degraded" health status

## Further Optimization Options

If you experience memory issues despite these optimizations:

1. Consider an even smaller embedding model
2. Implement API rate limiting
3. Add request timeout constraints
4. Consider upgrading to a higher tier Render plan
