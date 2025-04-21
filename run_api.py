#!/usr/bin/env python3

# Apply CUDA patch first, before any other imports
import os
# Force CPU-only mode before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

try:
    # Try to apply the CUDA patch if available
    from cuda_patch import patch_cuda
    patch_cuda()
except ImportError:
    # Create a simple patch function if the module is not available
    def patch_torch_cuda():
        import sys
        
        # Create a fake CUDA module
        class FakeCUDA:
            @staticmethod
            def is_available():
                return False
                
            @staticmethod
            def device_count():
                return 0
        
        # Patch torch.cuda if torch has been imported
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            
        # Install fake CUDA module
        sys.modules["torch.cuda"] = FakeCUDA()
    
    # Apply the simple patch
    patch_torch_cuda()

import argparse
import uvicorn
import sys
import logging
import time
from pathlib import Path

# Add the root directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import config and memory monitor
try:
    # Try to ensure psutil is installed
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not found, attempting to install...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil==5.9.5"])
        import psutil
        logger.info("psutil installed successfully")
    
    from src.config import (
        ENABLE_MEMORY_MONITORING, 
        DEFAULT_WORKERS,
        MAX_WORKERS,
        ON_RENDER
    )
    from scripts.memory_monitor import MemoryMonitor
    
    # Start memory monitoring if enabled
    if ENABLE_MEMORY_MONITORING:
        logger.info("Starting memory monitoring")
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
except ImportError as e:
    logger.warning(f"Could not import config or memory monitor: {e}")
    ENABLE_MEMORY_MONITORING = False
    DEFAULT_WORKERS = 1
    MAX_WORKERS = 4
    ON_RENDER = "RENDER" in os.environ

def main():
    parser = argparse.ArgumentParser(description="Run the Resume Scorer API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with auto-reload")
    parser.add_argument("--log-level", default="info", 
                      choices=["debug", "info", "warning", "error", "critical"],
                      help="Set logging level")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                       help=f"Number of worker processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--use-src", action="store_true", help="Use the src/api.py implementation instead of api/index.py")
    parser.add_argument("--use-render", action="store_true", help="Use the render_api/app.py implementation for Render deployment")
    parser.add_argument("--use-local", action="store_true", help="Use the local_api/app.py implementation for local development")
    parser.add_argument("--preload-models", action="store_true", help="Preload models before starting the server")
    parser.add_argument("--use-quantization", action="store_true", help="Enable model quantization (8-bit) to reduce memory usage")
    parser.add_argument("--task-specific-models", action="store_true", help="Use task-specific models for different parts of analysis")
    parser.add_argument("--optimize-memory", action="store_true", help="Apply aggressive memory optimizations")
    parser.add_argument("--fallback-to-cpu", action="store_true", help="Fallback to CPU if GPU is not available or fails")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip using ONNX runtime even if available")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for model loading (default: 3)")
    parser.add_argument("--safe-mode", action="store_true", help="Run in safe mode with minimal dependencies")
    
    args = parser.parse_args()
    
    # Set environment variables based on args
    if args.use_quantization:
        os.environ["USE_QUANTIZED_MODEL"] = "1"
        logger.info("Model quantization enabled (8-bit)")
        
    if args.task_specific_models:
        os.environ["USE_TASK_SPECIFIC_MODELS"] = "1"
        logger.info("Task-specific models enabled")
        
    if args.optimize_memory:
        os.environ["OPTIMIZE_MEMORY"] = "1"
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
        os.environ["PYTHONMALLOC"] = "malloc"
        os.environ["PYTORCH_JIT"] = "0"
        logger.info("Memory optimizations enabled")
    
    if args.fallback_to_cpu:
        os.environ["FALLBACK_TO_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("CPU fallback enabled")
    
    if args.skip_onnx:
        os.environ["SKIP_ONNX"] = "1"
        logger.info("ONNX runtime skipped")
    
    if args.retries:
        os.environ["MODEL_LOAD_RETRIES"] = str(args.retries)
        logger.info(f"Model load retries set to {args.retries}")
    
    if args.safe_mode:
        os.environ["SAFE_MODE"] = "1"
        logger.info("Safe mode enabled (minimal dependencies)")
    
    # Verify CUDA is properly disabled
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.warning("CUDA still showing as available! Forcing disable...")
            # Apply patch again to be sure
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
    except ImportError:
        logger.warning("Could not import torch to verify CUDA status")
    except Exception as e:
        logger.warning(f"Error checking CUDA status: {e}")
    
    # Cap workers based on environment
    if ON_RENDER and args.workers > MAX_WORKERS:
        logger.warning(f"Reducing workers from {args.workers} to {MAX_WORKERS} for Render compatibility")
        args.workers = MAX_WORKERS
    
    # Preload models if requested
    if args.preload_models or os.environ.get("MODEL_WARMUP_ON_START", "0") == "true":
        logger.info("Pre-loading models...")
        start_time = time.time()
        try:
            from src.utils.model_manager import warmup_models
            # Warm up the general model and skills-specific model
            warmup_models(["general", "skills"])
            logger.info(f"Models pre-loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error pre-loading models: {e}")
            logger.info("Continuing without pre-loaded models")
    
    # Determine which API implementation to use
    app_path = "api.index:app"  # Default to api/index.py for Render
    
    if args.use_render:
        app_path = "render_api.app:app"
        logger.info("Using Render API implementation")
    elif args.use_local:
        app_path = "local_api.app:app"
        logger.info("Using Local API implementation")
    elif args.use_src:
        app_path = "src.api:app"
        logger.info("Using Source API implementation")
    else:
        # Auto-detect based on environment
        if ON_RENDER:
            # Check if render_api directory exists
            if os.path.exists(os.path.join(current_dir, "render_api")):
                app_path = "render_api.app:app"
                logger.info("Auto-detected Render environment, using render_api.app:app")
            else:
                app_path = "api.index:app"
                logger.info("Auto-detected Render environment, falling back to api.index:app")
        else:
            # Check if local_api directory exists
            if os.path.exists(os.path.join(current_dir, "local_api")):
                app_path = "local_api.app:app"
                logger.info("Auto-detected local environment, using local_api.app:app")
            else:
                app_path = "api.index:app"
                logger.info("Auto-detected local environment, falling back to api.index:app")
    
    # Verify that the app path exists
    try:
        module_path, app_name = app_path.split(":")
        __import__(module_path)
        logger.info(f"Successfully imported {module_path}")
    except ImportError as e:
        logger.error(f"Could not import {module_path}: {e}")
        logger.info("Available modules:")
        for path in sys.path:
            logger.info(f" - {path}")
        logger.warning("Falling back to minimal API implementation")
        if os.path.exists(os.path.join(current_dir, "render_api", "app.py")):
            app_path = "render_api.app:app"
        else:
            # Create a minimal FastAPI app on the fly
            logger.info("Creating minimal API implementation")
            from fastapi import FastAPI
            app = FastAPI()
            
            @app.get("/")
            def root():
                return {"message": "API running in minimal mode due to import errors"}
            
            @app.get("/health")
            def health():
                return {"status": "healthy", "mode": "minimal"}
            
            # Save as the current module's attribute for uvicorn to find
            import __main__
            setattr(__main__, "app", app)
            app_path = "__main__:app"
    
    # Log startup information
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Using API implementation: {app_path}")
    logger.info(f"Workers: {args.workers}, Debug mode: {args.debug}")
    
    # Log optimization settings
    optimizations = []
    if args.use_quantization:
        optimizations.append("quantization")
    if args.task_specific_models:
        optimizations.append("task-specific models")
    if args.optimize_memory:
        optimizations.append("memory optimizations")
    if args.fallback_to_cpu:
        optimizations.append("CPU fallback")
    if args.skip_onnx:
        optimizations.append("ONNX skipped")
    if args.safe_mode:
        optimizations.append("safe mode")
    if optimizations:
        logger.info(f"Enabled optimizations: {', '.join(optimizations)}")
    
    if ON_RENDER:
        logger.info("Running on Render with optimized settings")
    
    # Run the API server
    try:
        uvicorn.run(
            app_path,
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level=args.log_level,
            workers=args.workers
        )
    except Exception as e:
        logger.error(f"Error starting uvicorn: {e}")
        
        # Try a different approach with a minimal app
        if app_path != "__main__:app":
            logger.info("Falling back to minimal server...")
            from fastapi import FastAPI
            app = FastAPI()
            
            @app.get("/")
            def root():
                return {"message": "Minimal API running due to startup errors"}
            
            @app.get("/health")
            def health():
                return {"status": "healthy", "mode": "recovery"}
            
            import __main__
            setattr(__main__, "app", app)
            
            uvicorn.run(
                "__main__:app",
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                workers=1  # Use just 1 worker for recovery mode
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
        # Stop memory monitoring if enabled
        if ENABLE_MEMORY_MONITORING and 'memory_monitor' in locals():
            memory_monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error running API server: {e}")
        sys.exit(1) 