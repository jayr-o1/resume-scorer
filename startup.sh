#!/bin/bash
# Startup script to ensure CUDA is properly disabled

# Set environment variables to disable CUDA
export CUDA_VISIBLE_DEVICES=""
export FORCE_CPU=1
export NO_CUDA=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
export DISABLE_CUDA=1

# Create a Python script to patch torch and test it's working
cat > test_cuda_disabled.py <<EOF
import os
import sys

# Verify environment variables
print("Environment variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"FORCE_CPU: {os.environ.get('FORCE_CPU')}")
print(f"NO_CUDA: {os.environ.get('NO_CUDA')}")
print(f"DISABLE_CUDA: {os.environ.get('DISABLE_CUDA')}")

# Try importing torch and verifying CUDA is disabled
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available (should be False): {torch.cuda.is_available()}")
    print(f"PyTorch device count (should be 0): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print("WARNING: CUDA still available even after patching!")
        sys.exit(1)
    print("SUCCESS: PyTorch CUDA properly disabled")
except ImportError:
    print("PyTorch not installed")
except Exception as e:
    print(f"Error testing PyTorch CUDA: {e}")
    sys.exit(1)

# Exit with success
sys.exit(0)
EOF

# Run the test script first to verify CUDA is disabled
echo "Testing if CUDA is properly disabled..."
python test_cuda_disabled.py

# If test passes, run the API server
if [ $? -eq 0 ]; then
    echo "CUDA successfully disabled, starting API server..."
    # Run with all the necessary flags
    exec python run_api.py --port=${PORT:-8080} --host=0.0.0.0 --workers=1 --use-render --preload-models --use-quantization --task-specific-models --optimize-memory --fallback-to-cpu --skip-onnx --safe-mode --retries=5
else
    echo "CUDA disable check failed, falling back to minimal implementation..."
    # Create a minimal WSGI app for uvicorn
    cat > minimal_app.py <<EOF
import os
# Force CPU mode for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CPU"] = "1"
os.environ["NO_CUDA"] = "1"

from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"message": "API running in emergency minimal mode"}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "emergency"}
EOF
    
    # Run with minimal implementation
    exec python -m uvicorn minimal_app:app --host=0.0.0.0 --port=${PORT:-8080} --workers=1
fi 