#!/usr/bin/env python3

# Apply CUDA patch first, before any other imports
import os
# Force CPU-only mode before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

# Simple CUDA patching function
def disable_cuda():
    """Disable CUDA functionality in PyTorch"""
    import sys
    
    class FakeCUDA:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def device_count():
            return 0
        
        # Add other methods that might be called
        @staticmethod
        def current_device():
            raise RuntimeError("CUDA not available")
        
        @staticmethod
        def get_device_name(device=None):
            return "CPU"
    
    # Patch sys.modules directly
    sys.modules["torch.cuda"] = FakeCUDA()
    
    # Patch torch if already imported
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0

# Try to apply the patch
try:
    disable_cuda()
except Exception as e:
    print(f"Warning: Failed to patch CUDA: {e}")

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
        MAX_WORKERS
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
    parser.add_argument("--preload-models", action="store_true", help="Preload models before starting the server")
    parser.add_argument("--use-quantization", action="store_true", help="Enable model quantization (8-bit) to reduce memory usage")
    parser.add_argument("--task-specific-models", action="store_true", help="Use task-specific models for different parts of analysis")
    parser.add_argument("--optimize-memory", action="store_true", help="Apply aggressive memory optimizations")
    parser.add_argument("--fallback-to-cpu", action="store_true", help="Fallback to CPU if GPU is not available or fails")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip using ONNX runtime even if available")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for model loading (default: 3)")
    
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
    
    # Verify CUDA is properly disabled
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
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
    if args.workers > MAX_WORKERS:
        logger.warning(f"Reducing workers from {args.workers} to {MAX_WORKERS} for compatibility")
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
    
    # Determine which API implementation to use - always use local_api
    app_path = "local_api.app:app"
    logger.info("Using Local API implementation")
    
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
        logger.error("Failed to load API module. Please check your installation.")
        sys.exit(1)
    
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
    if optimizations:
        logger.info(f"Enabled optimizations: {', '.join(optimizations)}")
    
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
        logger.error("API server failed to start. Please check your installation and configuration.")
        sys.exit(1)

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