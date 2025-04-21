#!/usr/bin/env python3
"""
Simple script to run the local API for testing.
This avoids the complexity of the main run_api.py script.
"""

import os
import uvicorn
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

if __name__ == "__main__":
    logger.info("Starting local API server...")
    
    # Verify that the local_api directory exists
    if not Path("local_api").exists() or not Path("local_api/app.py").exists():
        logger.error("local_api/app.py not found. Make sure you're in the correct directory.")
        exit(1)
    
    # Run the API server
    uvicorn.run(
        "local_api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 