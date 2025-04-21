#!/usr/bin/env python3
"""
Patch to disable CUDA functionality in PyTorch to avoid dependency issues.
This file should be preloaded or imported at the start of the application.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set environment variables to disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def patch_cuda():
    """Patch torch.cuda to prevent CUDA loading errors"""
    logger.info("Applying CUDA disabling patch")
    
    # Patch sys.modules directly to avoid any actual CUDA imports
    class FakeCUDA:
        """Fake CUDA module that returns False for availability checks"""
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def device_count():
            return 0
        
        @staticmethod
        def current_device():
            raise RuntimeError("CUDA not available")
        
        @staticmethod
        def get_device_name(device=None):
            return "CPU"
        
        class cudart:
            @staticmethod
            def cudaGetDevice(*args, **kwargs):
                raise RuntimeError("CUDA not available")
        
        # Mock the Stream object
        class Stream:
            null = None
            def __init__(self, *args, **kwargs):
                pass
            def wait_stream(self, *args, **kwargs):
                pass
    
    # Install the fake CUDA module
    sys.modules["torch.cuda"] = FakeCUDA()
    sys.modules["torch._C._cuda_isDriverSufficient"] = lambda: False
    
    # Patch torch if it's already been imported
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        torch.cuda.current_device = lambda: RuntimeError("CUDA not available")
        torch.cuda.get_device_name = lambda device=None: "CPU"
        
    logger.info("CUDA functionality disabled successfully")

# Apply the patch immediately
patch_cuda()

# Add a function to verify the patch worked
def verify_patch():
    """Verify that the CUDA patch has been applied correctly"""
    try:
        import torch
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        logger.info(f"PyTorch CUDA available: {available}, device count: {count}")
        assert not available, "CUDA should not be available"
        assert count == 0, "CUDA device count should be 0"
        return True
    except ImportError:
        logger.warning("Could not verify patch - torch not imported")
        return False
    except Exception as e:
        logger.error(f"Error verifying patch: {e}")
        return False

if __name__ == "__main__":
    # Configure logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Verify patch when run directly
    success = verify_patch()
    sys.exit(0 if success else 1) 