#!/bin/bash
# Startup script to ensure CUDA is properly disabled

# Set environment variables to disable CUDA
export CUDA_VISIBLE_DEVICES=""
export FORCE_CPU=1
export NO_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
export DISABLE_CUDA=1

# Try to import torch and patch CUDA
python -c "
import os
import sys

# Set environment variables that disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CPU'] = '1'
os.environ['NO_CUDA'] = '1'

try:
    import torch
    # Patch torch to disable CUDA
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    print('Patched torch.cuda successfully')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print(f'Error patching torch: {e}')
"

# Run the original API to maintain your setup
echo "Starting API with original implementation..."
python run_api.py --port=${PORT:-8080} --host=0.0.0.0 --workers=1 --use-render --fallback-to-cpu --skip-onnx 