#!/bin/bash
# Setup script for Flask Recall.ai app

echo "Setting up Flask Recall.ai application..."

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install requirements
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the application:"
echo "1. Activate virtual environment: source venv/Scripts/activate (Windows) or source venv/bin/activate (Linux/Mac)"
echo "2. Run: python app.py"
echo "3. Open: http://localhost:5000"
