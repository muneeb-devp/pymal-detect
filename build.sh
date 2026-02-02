#!/bin/bash
# Render Build Script
# This script runs during deployment on Render

set -e  # Exit on error

echo "======================================"
echo "Starting Render Build Process"
echo "======================================"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Python packages installed successfully!"

# Check if dataset exists
if [ ! -f "brazilian-malware-dataset-master/brazilian-malware.csv" ]; then
    echo "WARNING: Dataset not found!"
    echo "The dataset is too large for git. You need to:"
    echo "1. Upload it to external storage (S3, Google Drive, etc.)"
    echo "2. Download it during build, or"
    echo "3. Use a pre-trained model"
    echo ""
    echo "For now, creating models directory for pre-trained models..."
    mkdir -p models
else
    echo "Dataset found. Training model..."
    python train_for_deployment.py
fi

echo "======================================"
echo "Build Complete!"
echo "======================================"
