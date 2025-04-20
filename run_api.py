#!/usr/bin/env python3
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Run the Resume Scorer API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with auto-reload")
    parser.add_argument("--log-level", default="info", 
                      choices=["debug", "info", "warning", "error", "critical"],
                      help="Set logging level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--use-src", action="store_true", help="Use the src/api.py implementation instead of api/index.py")
    
    args = parser.parse_args()
    
    # Determine which API implementation to use
    app_path = "src.api:app" if args.use_src else "api.index:app"
    
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
    main() 