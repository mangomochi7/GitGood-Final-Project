#!/bin/bash
# Quick run script for Unix/Linux/Mac

echo "🚀 Starting GitGood Emotion Detection System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

# Check for API keys in .env
if grep -q "your_recall_api_key_here" .env; then
    echo "⚠️  WARNING: Please update your API keys in the .env file"
    echo "   - RECALL_API_KEY"
    echo "   - GEMINI_API_KEY"
    echo ""
fi

# Run the application
echo "✅ Starting application..."
echo "🌐 Web interface: http://localhost:5000"
echo "🔌 WebSocket: Port 3456"
echo ""
python app_simple.py
