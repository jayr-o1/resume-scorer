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
    
    args = parser.parse_args()
    
    # Project directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    scripts_dir = os.path.join(base_dir, "scripts")
    
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