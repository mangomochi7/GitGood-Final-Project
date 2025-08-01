@echo off
REM Quick run script for Windows

echo 🚀 Starting GitGood Emotion Detection System...

REM Check if virtual environment exists
if not exist venv (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env file exists and has API keys
if not exist .env (
    echo ❌ .env file not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Check for API keys in .env
findstr /C:"your_recall_api_key_here" .env >nul
if not errorlevel 1 (
    echo ⚠️  WARNING: Please update your API keys in the .env file
    echo    - RECALL_API_KEY
    echo    - GEMINI_API_KEY
    echo.
)

REM Run the application
echo ✅ Starting application...
echo 🌐 Web interface: http://localhost:5000
echo 🔌 WebSocket: Port 3456
echo.
python app_simple.py

pause
