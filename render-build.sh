#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies with optimized requirements for Render
pip install -r requirements-render.txt

# Download spaCy model if not using URL in requirements.txt
if ! pip show en-core-web-sm > /dev/null; then
  python -m spacy download en_core_web_sm
fi

# Create any necessary directories
mkdir -p ~/.streamlit

# Create Streamlit config
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
port = $PORT

[theme]
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
EOF 