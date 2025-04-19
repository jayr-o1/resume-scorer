#!/bin/bash

# Resume Scorer Deployment Script
# This script deploys the Resume Scorer to either AWS or GCP

set -e

# Default configuration
CLOUD_PROVIDER="aws"
SERVICE_NAME="resume-scorer"
REGION="us-east-1"
MEMORY="2048"
CPU="1"
DEPLOY_TYPE="serverless"  # or "container"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --provider)
      CLOUD_PROVIDER="$2"
      shift 2
      ;;
    --name)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --memory)
      MEMORY="$2"
      shift 2
      ;;
    --cpu)
      CPU="$2"
      shift 2
      ;;
    --type)
      DEPLOY_TYPE="$2"
      shift 2
      ;;
    --help)
      echo "Resume Scorer Deployment Script"
      echo "Usage: ./deploy.sh [options]"
      echo ""
      echo "Options:"
      echo "  --provider PROVIDER  - Cloud provider: aws or gcp (default: aws)"
      echo "  --name NAME          - Service name (default: resume-scorer)"
      echo "  --region REGION      - Deployment region (default: us-east-1 for AWS, us-central1 for GCP)"
      echo "  --memory MEMORY      - Memory allocation in MB (default: 2048)"
      echo "  --cpu CPU            - CPU allocation (default: 1)"
      echo "  --type TYPE          - Deployment type: serverless or container (default: serverless)"
      echo "  --help               - Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set region based on provider if not explicitly set
if [[ "$CLOUD_PROVIDER" == "gcp" && "$REGION" == "us-east-1" ]]; then
  REGION="us-central1"
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t "$SERVICE_NAME" .

# Set up credentials
if [[ "$CLOUD_PROVIDER" == "aws" ]]; then
  echo "Setting up AWS credentials..."
  
  # Check if AWS CLI is installed
  if ! command -v aws &> /dev/null; then
    echo "AWS CLI is required but not found. Please install it first."
    exit 1
  fi
  
  # Check if logged in
  if ! aws sts get-caller-identity &> /dev/null; then
    echo "Please log in to AWS first using 'aws configure'"
    exit 1
  fi
  
elif [[ "$CLOUD_PROVIDER" == "gcp" ]]; then
  echo "Setting up GCP credentials..."
  
  # Check if gcloud is installed
  if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK is required but not found. Please install it first."
    exit 1
  fi
  
  # Check if logged in
  if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "Please log in to GCP first using 'gcloud auth login'"
    exit 1
  fi
  
else
  echo "Unsupported cloud provider: $CLOUD_PROVIDER"
  echo "Supported providers: aws, gcp"
  exit 1
fi

# Deploy based on provider and type
if [[ "$CLOUD_PROVIDER" == "aws" ]]; then
  if [[ "$DEPLOY_TYPE" == "serverless" ]]; then
    echo "Deploying to AWS Lambda..."
    
    # Create serverless.yml if it doesn't exist
    if [[ ! -f "serverless.yml" ]]; then
      cat > serverless.yml << EOF
service: $SERVICE_NAME

provider:
  name: aws
  runtime: python3.10
  region: $REGION
  memorySize: $MEMORY
  timeout: 30

functions:
  api:
    handler: src/lambda_handler.handler
    events:
      - http: ANY /
      - http: ANY /{proxy+}
EOF
    fi
    
    # Install serverless framework if not available
    if ! command -v serverless &> /dev/null; then
      echo "Installing Serverless Framework..."
      npm install -g serverless
    fi
    
    # Create lambda handler
    mkdir -p src
    cat > src/lambda_handler.py << EOF
import json
from api import app
from mangum import Mangum

handler = Mangum(app)
EOF
    
    # Deploy
    echo "Deploying with Serverless Framework..."
    serverless deploy
    
  else
    echo "Deploying to AWS ECS..."
    
    # Authenticate with ECR
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$REGION.amazonaws.com"
    
    # Create repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names "$SERVICE_NAME" --region "$REGION" &> /dev/null; then
      aws ecr create-repository --repository-name "$SERVICE_NAME" --region "$REGION"
    fi
    
    # Tag and push the image
    REPO_URI="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$REGION.amazonaws.com/$SERVICE_NAME:latest"
    docker tag "$SERVICE_NAME" "$REPO_URI"
    docker push "$REPO_URI"
    
    echo "Image pushed to $REPO_URI"
    echo "Now create an ECS task definition and service using the AWS console or CLI..."
  fi
  
elif [[ "$CLOUD_PROVIDER" == "gcp" ]]; then
  if [[ "$DEPLOY_TYPE" == "serverless" ]]; then
    echo "Deploying to Google Cloud Run..."
    
    # Tag the image for Google Container Registry
    PROJECT_ID=$(gcloud config get-value project)
    IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
    
    docker tag "$SERVICE_NAME" "$IMAGE_NAME"
    
    # Push the image
    docker push "$IMAGE_NAME"
    
    # Deploy to Cloud Run
    gcloud run deploy "$SERVICE_NAME" \
      --image "$IMAGE_NAME" \
      --platform managed \
      --region "$REGION" \
      --memory "${MEMORY}Mi" \
      --cpu "$CPU" \
      --allow-unauthenticated
    
  else
    echo "Deploying to Google Kubernetes Engine..."
    
    # Tag the image for Google Container Registry
    PROJECT_ID=$(gcloud config get-value project)
    IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
    
    docker tag "$SERVICE_NAME" "$IMAGE_NAME"
    
    # Push the image
    docker push "$IMAGE_NAME"
    
    # Create a deployment YAML
    cat > deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $SERVICE_NAME
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $SERVICE_NAME
  template:
    metadata:
      labels:
        app: $SERVICE_NAME
    spec:
      containers:
      - name: $SERVICE_NAME
        image: $IMAGE_NAME
        resources:
          requests:
            memory: "${MEMORY}Mi"
            cpu: "${CPU}"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: $SERVICE_NAME
spec:
  selector:
    app: $SERVICE_NAME
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
EOF
    
    # Apply the deployment
    kubectl apply -f deployment.yaml
    
    echo "Deployed to Kubernetes. Check status with: kubectl get pods"
  fi
fi

echo "Deployment completed successfully!" 