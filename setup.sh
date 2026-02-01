#!/bin/bash
# Setup script for the Malware Detection System

echo "========================================="
echo "Malware Detection System - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p models
mkdir -p results
mkdir -p data
mkdir -p tests

# Create .gitkeep files
touch models/.gitkeep
touch results/.gitkeep
touch data/.gitkeep

echo ""
echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Train the models:"
echo "   python train.py --cv-folds 5"
echo ""
echo "3. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "4. Start the web application:"
echo "   python app.py"
echo ""
echo "========================================="
