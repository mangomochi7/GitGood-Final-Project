#!/bin/bash
# Quick setup for ML Processing System

echo "🚀 Setting up ML Processing System..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install flask flask-socketio requests python-dotenv pillow numpy

echo "✅ Setup complete!"
echo ""
echo "🎯 To run the system:"
echo "1. Set up ngrok: ngrok http 3456"
echo "2. Update YOUR_NGROK_URL in consolidated_ml_app.py"
echo "3. Run: python consolidated_ml_app.py"
echo "4. Open: http://localhost:5000"
echo ""
echo "📋 What this system does:"
echo "• Captures PNG frames from ALL meeting participants"
echo "• Sends clean PNG images to your ML team's models"
echo "• Includes participant metadata (name, ID, timestamp)"
echo "• Processes full transcripts with timestamps"
echo "• Provides messaging functionality"
echo "• Real-time web interface for monitoring"
