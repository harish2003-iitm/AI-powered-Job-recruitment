#!/bin/bash

echo "====================================="
echo "  TalentSphere AI Setup"
echo "====================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed."
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install backend dependencies."
    exit 1
fi
echo "Backend dependencies installed successfully."

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd frontend && npm install
if [ $? -ne 0 ]; then
    echo "Failed to install frontend dependencies."
    exit 1
fi
echo "Frontend dependencies installed successfully."

echo ""
echo "====================================="
echo "Setup completed successfully!"
echo ""
echo "To start the application, run:"
echo "  ./start.sh"
echo "=====================================" 