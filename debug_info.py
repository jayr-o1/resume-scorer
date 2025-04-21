import os
import sys
import importlib
import traceback
import platform
import socket
import datetime
import json

def print_debug_info():
    """Print debug information about the environment."""
    print("=" * 50)
    print("DEBUG INFO")
    print("=" * 50)
    
    # System information
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Current time: {datetime.datetime.now().isoformat()}")
    
    # Python path and current directory
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    
    cwd = os.getcwd()
    print(f"\nCurrent working directory: {cwd}")
    
    # Directory contents
    print("\nDirectory contents:")
    try:
        for item in sorted(os.listdir(cwd)):
            if os.path.isdir(os.path.join(cwd, item)):
                print(f"  - {item}/ (directory)")
            else:
                print(f"  - {item} (file)")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Check for specific directories
    for directory in ["src", "local_api"]:
        dir_path = os.path.join(cwd, directory)
        print(f"\nChecking {directory} directory:")
        if os.path.exists(dir_path):
            print(f"  - {directory} exists")
            print(f"  - Contents:")
            try:
                for item in sorted(os.listdir(dir_path)):
                    if os.path.isdir(os.path.join(dir_path, item)):
                        print(f"    - {item}/ (directory)")
                    else:
                        print(f"    - {item} (file)")
            except Exception as e:
                print(f"  - Error listing {directory}: {e}")
        else:
            print(f"  - {directory} does not exist")
    
    # Try importing key modules
    print("\nTrying to import key modules:")
    modules_to_check = [
        "local_api.app", 
        "src", 
        "src.utils",
        "src.utils.model_manager"
    ]
    
    for module_name in modules_to_check:
        print(f"  - Importing {module_name}... ", end="")
        try:
            module = importlib.import_module(module_name)
            print("SUCCESS")
            if module_name == "local_api.app":
                print(f"    - app object exists: {'app' in dir(module)}")
        except Exception as e:
            print("FAILED")
            print(f"    - Error: {e}")
            print(f"    - Traceback:")
            traceback.print_exc(limit=3)
    
    # Check installed packages
    try:
        import pkg_resources
        print("\nInstalled packages:")
        for pkg in sorted([f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set]):
            print(f"  - {pkg}")
    except ImportError:
        print("\nCould not import pkg_resources to list installed packages")
    
    # Environment variables (excluding sensitive info)
    print("\nEnvironment variables (excluding potential secrets):")
    safe_vars = {k: v for k, v in os.environ.items() 
                if not any(x in k.lower() for x in ['key', 'secret', 'token', 'password', 'auth'])}
    for k, v in sorted(safe_vars.items()):
        if len(v) > 50:
            v = v[:47] + "..."
        print(f"  - {k}: {v}")
    
    # Hardware info
    try:
        import psutil
        print("\nHardware resources:")
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        print(f"  - CPU cores: {cpu_count}")
        print(f"  - RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
        print(f"  - Disk: {psutil.disk_usage('/').free / (1024**3):.2f} GB free of {psutil.disk_usage('/').total / (1024**3):.2f} GB")
    except ImportError:
        print("\nCould not import psutil to get hardware information")
    
    # Check for CUDA / GPU
    try:
        import torch
        print("\nTorch version and CUDA availability:")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\nCould not import torch to check CUDA availability")
    
    print("=" * 50)

if __name__ == "__main__":
    print_debug_info() 