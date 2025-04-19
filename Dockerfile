FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p model_cache src/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8501

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
  cd src && python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}\n\
elif [ "$1" = "streamlit" ]; then\n\
  cd src && streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0\n\
else\n\
  cd src && python run.py "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["streamlit"] 