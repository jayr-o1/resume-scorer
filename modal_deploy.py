import modal
from modal_app import app, download_models

if __name__ == "__main__":
    # Deploy the application to Modal
    print("Deploying the Resume Scorer API to Modal...")
    
    # Deploy the app using the Modal CLI
    import subprocess
    subprocess.run(["modal", "deploy", "modal_app.py"])
    
    # Run the download_models function after deployment
    print("Downloading and caching models...")
    try:
        modal.Function.from_name("resume-scorer", "download_models").remote()
        print("Models downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("You may need to manually run: modal run modal_app.py::download_models")
    
    print("Resume Scorer API deployed successfully!")
    print("You can access your API at: https://resume-scorer--fastapi-app.modal.run") 