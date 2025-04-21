#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the AI Resume Scorer application")
    parser.add_argument("--mode", choices=["web", "web-local", "cli", "cli-local", "pdf"], default="web-local",
                        help="Run mode: web (requires API), web-local (no API), cli (requires API), cli-local (no API), or pdf creator")
    parser.add_argument("--pdf_input", help="Input file for PDF creation (for pdf mode)")
    parser.add_argument("--pdf_output", help="Output file for PDF creation (for pdf mode)")
    parser.add_argument("--resume", help="Resume PDF path (for cli/cli-local mode)")
    parser.add_argument("--job_file", help="Job details JSON file (for cli/cli-local mode)")
    parser.add_argument("--use-quantization", action="store_true", help="Enable model quantization (8-bit) to reduce memory usage")
    parser.add_argument("--task-specific-models", action="store_true", help="Use task-specific models for different parts of analysis")
    parser.add_argument("--optimize-memory", action="store_true", help="Apply aggressive memory optimizations")
    
    args = parser.parse_args()
    
    # Project directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # Set environment variables based on args
    if args.use_quantization:
        os.environ["USE_QUANTIZED_MODEL"] = "1"
        print("Model quantization enabled (8-bit)")
        
    if args.task_specific_models:
        os.environ["USE_TASK_SPECIFIC_MODELS"] = "1"
        print("Task-specific models enabled")
        
    if args.optimize_memory:
        os.environ["OPTIMIZE_MEMORY"] = "1"
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
        os.environ["PYTHONMALLOC"] = "malloc"
        os.environ["PYTORCH_JIT"] = "0"
        print("Memory optimizations enabled")
    
    # Run based on mode
    if args.mode == "web":
        # Run the web interface with OpenAI
        os.chdir(src_dir)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    
    elif args.mode == "web-local":
        # Run the web interface with local model (no API needed)
        os.chdir(src_dir)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_local.py"])
    
    elif args.mode == "cli":
        # Run the command line interface with OpenAI
        if not args.resume:
            print("Error: Resume PDF path is required for CLI mode")
            return
        
        os.chdir(src_dir)
        cmd = [sys.executable, "cli.py", args.resume]
        
        if args.job_file:
            cmd.extend(["--job_file", args.job_file])
        
        subprocess.run(cmd)
    
    elif args.mode == "cli-local":
        # Run the command line interface with local model (no API needed)
        if not args.resume:
            print("Error: Resume PDF path is required for CLI-local mode")
            return
        
        os.chdir(src_dir)
        cmd = [sys.executable, "cli_local.py", args.resume]
        
        if args.job_file:
            cmd.extend(["--job_file", args.job_file])
        
        subprocess.run(cmd)
    
    elif args.mode == "pdf":
        # Run the PDF creator
        os.chdir(scripts_dir)
        cmd = [sys.executable, "create_pdf.py"]
        
        if args.pdf_input and args.pdf_output:
            cmd.extend([args.pdf_input, args.pdf_output])
        
        subprocess.run(cmd)

if __name__ == "__main__":
    main() 