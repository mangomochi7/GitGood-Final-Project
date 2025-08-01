#!/bin/bash
# Comprehensive Setup Script for GitGood Recall.ai Emotion Detection System

echo "🚀 Setting up GitGood Recall.ai Emotion Detection System..."
echo "=================================================="

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python --version 2>&1)
echo "Found: $python_version"

# Check if Python 3.8+ is available
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ ERROR: Python 3.8+ is required. Please install Python 3.8 or higher."
    echo "   Download from: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python version is compatible"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python -m venv venv

# Activate virtual environment based on OS
echo ""
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
    echo "Windows environment detected"
else
    source venv/bin/activate
    echo "Unix/Linux/Mac environment detected"
fi

# Upgrade pip first
echo ""
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing Python packages..."
echo "This may take several minutes for first-time setup..."

# Install packages in order of importance (core dependencies first)
echo "Installing core web framework..."
pip install Flask==2.3.3 Flask-SocketIO==5.3.6

echo "Installing communication libraries..."
pip install websockets==12.0 requests==2.31.0 python-dotenv==1.0.0

echo "Installing computer vision libraries..."
pip install opencv-python==4.8.1.78 Pillow==10.0.1 numpy==1.24.3

echo "Installing AI/ML libraries (this may take longer)..."
pip install deepface==0.0.79 mediapipe==0.10.8

echo "Installing Google Generative AI..."
pip install google-generativeai

echo "Installing remaining dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p transcripts
mkdir -p processed_frames
mkdir -p screenshots
mkdir -p templates
mkdir -p static

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'
# Recall.ai API Configuration
RECALL_API_KEY=your_recall_api_key_here

# Google Generative AI Configuration  
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
FLASK_PORT=5000
WEBSOCKET_PORT=3456

# Development Settings
FLASK_DEBUG=False
EOF
    echo "✅ Created .env template - PLEASE UPDATE WITH YOUR API KEYS!"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "
import flask
import cv2
import numpy
import PIL
import deepface
import mediapipe
print('✅ All core packages imported successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "=================================================="
    echo ""
    echo "📋 Next Steps:"
    echo "1. 🔑 Update .env file with your API keys:"
    echo "   - Get Recall.ai API key from: https://recall.ai"
    echo "   - Get Google Gemini API key from: https://makersuite.google.com"
    echo ""
    echo "2. 🌐 Set up ngrok for webhook connectivity:"
    echo "   - Install ngrok: https://ngrok.com/download"
    echo "   - Run: ngrok http 3456"
    echo "   - Update the ngrok URL in app_simple.py (line ~300)"
    echo ""
    echo "3. 🚀 Start the application:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "   - Activate environment: source venv/Scripts/activate"
    else
        echo "   - Activate environment: source venv/bin/activate"
    fi
    echo "   - Run: python app_simple.py"
    echo "   - Open: http://localhost:5000"
    echo ""
    echo "4. 📺 Test with a meeting:"
    echo "   - Join a Zoom/Google Meet/Teams meeting"
    echo "   - Send the bot via the web interface"
    echo "   - Watch emotion detection in the terminal!"
    echo ""
    echo "=================================================="
else
    echo ""
    echo "❌ Setup encountered issues. Please check the error messages above."
    echo "💡 Common solutions:"
    echo "   - Ensure you have Python 3.8+"
    echo "   - Try: pip install --upgrade pip"
    echo "   - For M1 Macs: pip install tensorflow-metal"
    echo "   - For GPU support: pip install tensorflow-gpu"
fi
