# Deploying Resume Scorer on Render

This guide will help you deploy the Resume Scorer application on Render.

## Prerequisites

- A [Render account](https://render.com/)
- Your code pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

1. Log in to your Render account
2. Click "New" and select "Web Service"
3. Connect your Git repository
4. Configure the service with the following settings:

   - **Name**: resume-scorer (or your preferred name)
   - **Environment**: Python
   - **Region**: Choose the closest to your users
   - **Branch**: main (or your default branch)
   - **Build Command**: `bash render-build.sh`
   - **Start Command**: `python -m streamlit run src/app.py`
   - **Plan**: Free (or select a paid plan if you need more resources)

5. Click "Create Web Service"

## Environment Variables

The following environment variables are set in the `render.yaml` configuration:

- `PYTHON_VERSION`: 3.10.0
- `PORT`: 8501

If your application requires any additional environment variables (API keys, database credentials, etc.), add them in the Render dashboard under the "Environment" section.

## Important Notes

1. The free tier of Render has the following limitations:
   - Spins down with inactivity
   - 750 hours free per month
   - Shared CPU
   - 512 MB RAM
   - 0.5 GB persistent disk

2. If your application requires more resources, consider upgrading to a paid plan.

3. Large machine learning models may need to be downloaded during build time, which could increase build times.

## Optimizations for Render

To stay within Render's free tier limits and reduce the size of the application, the following optimizations have been made:

1. **CPU-only PyTorch**: We use the CPU-only version of PyTorch to significantly reduce the size of dependencies.
2. **Optimized Requirements**: A separate `requirements-render.txt` file is used for deployment which includes optimized versions of the dependencies.
3. **Persistent Disk**: A small 1GB persistent disk is configured to store temporary files and cached models.

## Additional Configuration

If your application requires storage for uploaded files or model caches, you can use the `/data` directory which is mounted as a persistent disk.

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs in Render dashboard
2. Ensure your application is compatible with Python 3.10
3. Verify that all dependencies are correctly specified in requirements.txt
4. Check that your Streamlit app can run on port 8501 