#!/usr/bin/env python3
"""
CLEAN MINIMAL VERSION - Recall.ai + Emotion Detection
Focus: Get video frames -> Detect emotions -> Print results
No bloat, just working code.
"""

import os
import json
import base64
import socket
import hashlib
import struct
import threading
import zlib  # Add zlib for decompression
import asyncio
import websockets  # Use proper WebSocket library like Node.js ws
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import requests
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import io

# Try to import emotion detection
try:
    from emotion_by_images import get_emotion
    print("✅ Emotion detection loaded")
except ImportError:
    print("⚠️  Using placeholder emotions")
    def get_emotion(frame):
        import random
        return [random.choice(['happy', 'neutral', 'sad', 'angry', 'surprise'])]

# Configuration
RECALL_API_KEY = os.getenv('RECALL_API_KEY', '11e4159cb943ded9085f5a8f1ab691c21af45d67')  # Get from env or use default
WEBSOCKET_PORT = 3456
FLASK_PORT = 5000

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class SimpleEmotionDetector:
    def __init__(self):
        self.participants = {}
    
    def process_frame(self, participant_id, participant_name, image_array):
        """Process a video frame and detect emotions"""
        try:
            print(f"\n🎯 PROCESSING: {participant_name}")
            print(f"   📊 Image: {image_array.shape} | {image_array.dtype}")
            
            # Ensure we have RGB format (3 channels) - should already be converted upstream
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                print(f"   ⚠️  Expected RGB format, got: {image_array.shape}")
                return 'error'
            
            # Convert RGB to BGR for OpenCV (OpenCV expects BGR format)
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Detect emotions
            emotions = get_emotion(bgr_image)
            primary_emotion = emotions[0] if emotions else 'neutral'
            
            # Store participant data
            self.participants[participant_id] = {
                'name': participant_name,
                'emotion': primary_emotion,
                'last_updated': datetime.now()
            }
            
            # Print results clearly
            print(f"   🎭 EMOTION: {primary_emotion.upper()}")
            print(f"   👥 Total participants: {len(self.participants)}")
            print("-" * 50)
            
            # Emit real-time update via SocketIO
            try:
                from flask_socketio import emit
                socketio.emit('emotion_update', {
                    'participant_id': participant_id,
                    'participant_name': participant_name,
                    'emotion': primary_emotion,
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass  # SocketIO not available
            
            return primary_emotion
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return 'error'

# Initialize detector
detector = SimpleEmotionDetector()

def handle_recall_message(message_data):
    """Handle incoming message from Recall.ai"""
    try:
        event = message_data.get('event')
        print(f"📨 Received event: {event}")
        
        if event == "video_separate_png.data":
            # Extract participant info
            participant_info = message_data['data']['data']['participant']
            participant_id = participant_info.get('id')
            participant_name = participant_info.get('name') or f"User_{participant_id}"
            
            # Get base64 image data
            base64_image = message_data['data']['data']['buffer']
            
            # Convert base64 to image
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Handle RGBA (4 channels) by converting to RGB (3 channels)
            if image_array.shape[2] == 4:
                print(f"   📷 Converting RGBA to RGB (removing alpha channel)")
                # Convert RGBA to RGB by removing alpha channel
                image_array = image_array[:, :, :3]
            
            # Process the frame
            emotion = detector.process_frame(participant_id, participant_name, image_array)
            
            print(f"✅ SUCCESS: {participant_name} -> {emotion}")
            
        elif event in ["transcript.data", "transcript.partial_data"]:
            print(f"📝 Transcript received: {event}")
            
            # DEBUG: Print actual transcript data structure
            try:
                transcript_data = message_data['data']['data']
                print(f"🔍 DEBUG - Transcript data keys: {list(transcript_data.keys())}")
                print(f"🔍 DEBUG - Full transcript data: {transcript_data}")
            except:
                print(f"🔍 DEBUG - Full message: {message_data}")
            
            # Save transcript to file
            try:
                # Create transcripts directory if it doesn't exist
                transcripts_dir = "transcripts"
                if not os.path.exists(transcripts_dir):
                    os.makedirs(transcripts_dir)
                    print(f"📁 Created transcripts directory: {transcripts_dir}")
                
                # Extract transcript data - try different possible field names
                transcript_data = message_data['data']['data']
                transcript_text = None
                
                # Try common field names for transcript text
                for field_name in ['text', 'transcript', 'content', 'words', 'speech']:
                    if field_name in transcript_data:
                        transcript_text = transcript_data[field_name]
                        print(f"✅ Found transcript text in field: '{field_name}'")
                        break
                
                if transcript_text is None:
                    print(f"⚠️ No transcript text found in available fields: {list(transcript_data.keys())}")
                    return
                
                # Extract participant and timestamp info
                participant_info = transcript_data.get('participant', {})
                participant_name = participant_info.get('name') or f"User_{participant_info.get('id', 'unknown')}"
                
                # Try to get timestamp
                timestamp_info = transcript_data.get('timestamp', {})
                timestamp_str = timestamp_info.get('absolute', datetime.now().isoformat()) if timestamp_info else datetime.now().isoformat()
                
                # Use ONE big transcript file (not multiple files)
                filename = "meeting_transcript.txt"
                filepath = os.path.join(transcripts_dir, filename)
                
                # Format transcript entry with event type info
                event_marker = "🔴 FINAL" if event == "transcript.data" else "⚪ PARTIAL"
                entry = f"[{timestamp_str}] {event_marker} {participant_name}: {transcript_text}\n"
                
                # Append to the single big transcript file
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(entry)
                
                print(f"💾 Saved to: {filepath}")
                print(f"   📝 {participant_name}: {str(transcript_text)[:50]}...")
                
            except Exception as e:
                print(f"❌ Error saving transcript: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error handling message: {e}")
        import traceback
        traceback.print_exc()

async def handle_websocket_client(websocket):
    """Handle WebSocket connection from Recall.ai - mimics Node.js ws library functionality"""
    print(f"🎉 NEW CONNECTION! Recall.ai bot connected from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # The websockets library automatically handles all the protocol details!
                # Just like Node.js: const wsMessage = JSON.parse(message.toString())
                json_data = json.loads(message)
                event_type = json_data.get('event', 'unknown')
                print(f"📨 Received event: {event_type}")
                
                # Process the message (same as Node.js TypeScript code)
                handle_recall_message(json_data)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse JSON: {e}")
                print(f"   Raw message: {message[:100]}...")
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Connection closed by {websocket.remote_address}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print(f"🔌 Disconnected: {websocket.remote_address}")

def start_websocket_server():
    """Start WebSocket server using proper websockets library (like Node.js ws)"""
    async def run_server():
        print(f"🔌 WebSocket server listening on port {WEBSOCKET_PORT}")
        print(f"🌐 Expecting connections from Recall.ai bot")
        print(f"⏳ Waiting for video frames and emotions...")
        
        # Start WebSocket server - automatically handles all protocol details
        server = await websockets.serve(
            handle_websocket_client,
            "0.0.0.0",
            WEBSOCKET_PORT
        )
        
        # Keep server running
        await server.wait_closed()
    
    def run_event_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
    
    thread = threading.Thread(target=run_event_loop, daemon=True)
    thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send-bot', methods=['POST'])
def send_bot():
    """Send bot to meeting - using app_ml.py logic"""
    data = request.get_json()
    meeting_url = data.get('meeting_url')
    bot_name = data.get('bot_name')
    recording_config = data.get('recording_config')
    
    print(f"\n🤖 RECALL.AI BOT REQUEST RECEIVED")
    print(f"   Meeting URL: {meeting_url}")
    print(f"   Bot Name: {bot_name}")
    print(f"   Recording Config: {recording_config}")
    print(f"   API Key Present: {'✅ YES' if RECALL_API_KEY else '❌ NO'}")
    
    if not meeting_url:
        error_msg = "Error in /send-bot: meeting_url is required"
        print(f"❌ {error_msg}")
        return jsonify({'error': 'meeting_url is required'}), 400
    
    if not RECALL_API_KEY:
        error_msg = "Error in /send-bot: RECALL_API_KEY is not set in server environment"
        print(f"❌ {error_msg}")
        return jsonify({'error': 'RECALL_API_KEY is not set in environment variables'}), 500
    
    try:
        recall_api_url = "https://us-west-2.recall.ai/api/v1/bot"
        
        # Build payload - use provided recording_config or fallback to default
        payload = {
            'meeting_url': meeting_url,
            'bot_name': bot_name or 'Emotion Bot'
        }
        
        # If recording_config is provided by frontend, use it
        if recording_config:
            payload['recording_config'] = recording_config
            print(f"📋 Using frontend recording config with events: {recording_config.get('realtime_endpoints', [{}])[0].get('events', [])}")
        else:
            # Fallback to DEFAULT config matching app_ml.py EXACTLY
            payload['recording_config'] = {
                'realtime_endpoints': [
                    {
                        'url': 'wss://76da8c795ea9.ngrok-free.app',  # Current active ngrok URL
                        'events': [
                            'video_separate_png.data',
                            'transcript.data'  # Only full transcripts for clean meeting notes
                        ]
                    }
                ]
            }
            print(f"📋 Using default recording config")
        
        print(f"📤 Sending request to Recall.ai API...")
        
        response = requests.post(
            recall_api_url,
            json=payload,
            headers={
                'Authorization': f'Token {RECALL_API_KEY}',
                'Content-Type': 'application/json'
            },
            timeout=30
        )
        
        print(f"📨 Recall.ai API Response: {response.status_code}")
        
        if response.status_code == 200 or response.status_code == 201:
            success_msg = f"✅ Successfully sent bot to meeting!"
            print(success_msg)
            return jsonify(response.json()), response.status_code
        else:
            error_msg = response.json() if response.content else response.text
            print(f"❌ Recall.ai API Error: {response.status_code} - {error_msg}")
            print(f"❌ Response body: {response.text}")
            try:
                error_data = response.json()
                print(f"❌ Error details: {error_data}")
            except:
                pass
            return jsonify(error_msg), response.status_code
            
    except Exception as error:
        error_msg = str(error)
        print(f"❌ Error calling Recall.ai API: {error_msg}")
        return jsonify({'error': 'Failed to send bot to meeting due to an internal server error.'}), 500

@app.route('/participants')
def get_participants():
    """Get current participants and their emotions"""
    return jsonify({
        'participants': detector.participants,
        'total': len(detector.participants)
    })

# SocketIO event handlers for the original HTML template
@socketio.on('connect')
def handle_connect():
    print(f"🔌 Client connected via SocketIO")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected from SocketIO")

@socketio.on('get_participants')
def handle_get_participants():
    """Send participant data via SocketIO"""
    emit('participants_update', {
        'participants': detector.participants,
        'total': len(detector.participants)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SIMPLE RECALL.AI EMOTION DETECTOR")
    print("="*60)
    print(f"🌐 Web UI: http://localhost:{FLASK_PORT}")
    print(f"🔌 WebSocket: Port {WEBSOCKET_PORT}")
    print(f"🤖 Recall.ai API: Ready")
    print("="*60)
    print("📋 INSTRUCTIONS:")
    print("1. Make sure ngrok is running: ngrok http 3456")
    print("2. Update websocket_url in send_bot() with your ngrok URL")
    print("3. Open web UI and send bot to meeting")
    print("4. Watch emotions appear in this terminal!")
    print("="*60 + "\n")
    
    # Start WebSocket server
    start_websocket_server()
    
    # Start Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=FLASK_PORT, debug=False)
