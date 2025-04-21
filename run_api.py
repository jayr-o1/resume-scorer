#!/usr/bin/env python3
import argparse
import uvicorn
import os
import sys
import logging

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
    
    args = parser.parse_args()
    
    # Cap workers based on environment
    if ON_RENDER and args.workers > MAX_WORKERS:
        logger.warning(f"Reducing workers from {args.workers} to {MAX_WORKERS} for Render compatibility")
        args.workers = MAX_WORKERS
    
    # Preload models if requested
    if args.preload_models:
        logger.info("Preloading models...")
        try:
            from scripts.download_models import download_sentence_transformer, download_spacy_model
            download_sentence_transformer()
            download_spacy_model()
            logger.info("Models preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")
    
    # Determine which API implementation to use
    app_path = "src.api:app"  # Default to src/api.py
    
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
        logger.warning("Falling back to api.index:app")
        app_path = "api.index:app"
    
    # Log startup information
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Using API implementation: {app_path}")
    logger.info(f"Workers: {args.workers}, Debug mode: {args.debug}")
    
    if ON_RENDER:
        logger.info("Running on Render with optimized settings")
    
    # Run the API server
    uvicorn.run(
        app_path,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level=args.log_level,
        workers=args.workers
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