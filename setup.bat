@echo off
REM Comprehensive Setup Script for GitGood Recall.ai Emotion Detection System (Windows)

echo 🚀 Setting up GitGood Recall.ai Emotion Detection System...
echo ==================================================

REM Check Python version
echo 🐍 Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH.
    echo    Please install Python 3.8+ from: https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check if Python 3.8+ is available
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Python 3.8+ is required. Please install Python 3.8 or higher.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python version is compatible

REM Create virtual environment
echo.
echo 📦 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip first
echo.
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo 📥 Installing Python packages...
echo This may take several minutes for first-time setup...

REM Install packages in order of importance
echo Installing core web framework...
pip install Flask==2.3.3 Flask-SocketIO==5.3.6

echo Installing communication libraries...
pip install websockets==12.0 requests==2.31.0 python-dotenv==1.0.0

echo Installing computer vision libraries...
pip install opencv-python==4.8.1.78 Pillow==10.0.1 numpy==1.24.3

echo Installing AI/ML libraries (this may take longer)...
pip install deepface==0.0.79 mediapipe==0.10.8

echo Installing Google Generative AI...
pip install google-generativeai

echo Installing remaining dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo 📁 Creating project directories...
if not exist transcripts mkdir transcripts
if not exist processed_frames mkdir processed_frames
if not exist screenshots mkdir screenshots
if not exist templates mkdir templates
if not exist static mkdir static

REM Create .env template if it doesn't exist
if not exist .env (
    echo.
    echo 📝 Creating .env template...
    (
        echo # Recall.ai API Configuration
        echo RECALL_API_KEY=your_recall_api_key_here
        echo.
        echo # Google Generative AI Configuration
        echo GEMINI_API_KEY=your_gemini_api_key_here
        echo.
        echo # Server Configuration
        echo FLASK_PORT=5000
        echo WEBSOCKET_PORT=3456
        echo.
        echo # Development Settings
        echo FLASK_DEBUG=False
    ) > .env
    echo ✅ Created .env template - PLEASE UPDATE WITH YOUR API KEYS!
)

REM Verify installation
echo.
echo 🔍 Verifying installation...
python -c "import flask; import cv2; import numpy; import PIL; import deepface; import mediapipe; print('✅ All core packages imported successfully!')" 2>nul

if errorlevel 1 (
    echo.
    echo ❌ Setup encountered issues. Please check the error messages above.
    echo 💡 Common solutions:
    echo    - Ensure you have Python 3.8+
    echo    - Try: pip install --upgrade pip
    echo    - Close and reopen command prompt
    echo    - Run as Administrator if needed
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo ==================================================
echo.
echo 📋 Next Steps:
echo 1. 🔑 Update .env file with your API keys:
echo    - Get Recall.ai API key from: https://recall.ai
echo    - Get Google Gemini API key from: https://makersuite.google.com
echo.
echo 2. 🌐 Set up ngrok for webhook connectivity:
echo    - Install ngrok: https://ngrok.com/download
echo    - Run: ngrok http 3456
echo    - Update the ngrok URL in app_simple.py (around line 300)
echo.
echo 3. 🚀 Start the application:
echo    - Double-click: run.bat
echo    - OR manually: venv\Scripts\activate.bat then python app_simple.py
echo    - Open: http://localhost:5000
echo.
echo 4. 📺 Test with a meeting:
echo    - Join a Zoom/Google Meet/Teams meeting
echo    - Send the bot via the web interface
echo    - Watch emotion detection in the terminal!
echo.
echo ==================================================
pause
