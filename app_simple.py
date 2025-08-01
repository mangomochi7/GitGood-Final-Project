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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
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
from main import *
from posture_database import database
from emotion_by_images import get_emotion
from emotion_by_images import EmotionCounter
import google.generativeai as genai

# Configuration
RECALL_API_KEY = os.getenv('RECALL_API_KEY', '11e4159cb943ded9085f5a8f1ab691c21af45d67')  # Get from env or use default
WEBSOCKET_PORT = 3456
FLASK_PORT = 5000

# Global variable to store bot_id from Recall.ai API response
bot_id = None

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

db = database()
counter = EmotionCounter()

class SimpleEmotionDetector:
    def __init__(self):
        self.participants = {}
    
    def process_frame(self, participant_id, participant_name, image_array, frame_timestamp=None):
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
            primary_emotion = get_emotion(bgr_image, counter)
            if counter.is_triggered():
                context_window = get_transcript_context(frame_timestamp)
                mess = trigger_llm_call(context_window, "Feeling a bad emotion", participant_name)
                send_zoom_message(mess, bot_id)

            # Analyze posture
            posture(bgr_image, db, participant_name, trigger_llm_call)

            # Store participant data
            self.participants[participant_id] = {
                'name': participant_name,
                'emotion': primary_emotion,
                'posture_status': db.get_main_status(),
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
            
            frame_timestamp = message_data['data']['data'].get('timestamp', datetime.now().isoformat())

            # Process the frame
            emotion = detector.process_frame(participant_id, participant_name, image_array, frame_timestamp=frame_timestamp)
            
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
            
            # Capture bot_id from response
            global bot_id
            response_data = response.json()
            bot_id = response_data.get('id')
            print(f"🤖 Bot ID captured: {bot_id}")
            
            return jsonify(response_data), response.status_code
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

def get_transcript_context(persistent_emotion_time_stamps, context_length=120):
    """
    Get transcript context around persistent emotion timestamps
    Parses the JSON format in your transcript file
    timestamp example: 2025-07-31T13:50:40
    """
    try:
        transcript_file = "transcripts/meeting_transcript.txt"
        
        if not os.path.exists(transcript_file):
            print("⚠️ No transcript file found")
            return {}
        
        # Read all transcript entries
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_lines = f.readlines()
        
        context_windows = {}
        
        for emotion_timestamp in persistent_emotion_time_stamps:
            print(f"🔍 Getting context for emotion at: {emotion_timestamp}")
            
            relevant_entries = []
            
            for line in transcript_lines:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse the line format: [timestamp] 🔴 FINAL Name: [JSON data]
                    if '[' in line and '] 🔴 FINAL' in line:
                        # Extract the timestamp from the beginning: [2025-07-31T13:50:19.388127]
                        timestamp_str = line.split(']')[0][1:]  # Remove [ and ]
                        transcript_time = datetime.fromisoformat(timestamp_str)
                        
                        # Extract speaker name: between "FINAL " and ": "
                        name_part = line.split('🔴 FINAL ')[1].split(': ')[0]
                        
                        # Extract and parse the JSON data
                        json_part = line.split(': [', 1)[1]  # Everything after ": ["
                        json_data = json.loads('[' + json_part)  # Add back the opening [
                        
                        # Extract the actual text from the JSON
                        if json_data and isinstance(json_data, list) and len(json_data) > 0:
                            text_content = json_data[0].get('text', '')
                            
                            # Check if this transcript is within our context window
                            time_diff = abs((emotion_timestamp - transcript_time).total_seconds())
                            
                            if time_diff <= context_length:
                                relevant_entries.append({
                                    'timestamp': transcript_time,
                                    'speaker': name_part,
                                    'text': text_content,
                                    'time_diff': time_diff
                                })
                                
                except Exception as e:
                    print(f"⚠️ Could not parse line: {line[:50]}... Error: {e}")
                    continue
            
            # Sort by timestamp
            relevant_entries.sort(key=lambda x: x['timestamp'])
            
            # Create clean context text
            context_lines = []
            for entry in relevant_entries:
                time_str = entry['timestamp'].strftime('%H:%M:%S')
                context_lines.append(f"[{time_str}] {entry['speaker']}: {entry['text']}")
            
            context_text = '\n'.join(context_lines)
            context_windows[emotion_timestamp.isoformat()] = context_text
            
            print(f"📝 Found {len(relevant_entries)} relevant transcript entries")
        
        # Save context windows to file
        context_file = "context_windows.txt"
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write("=== EMOTION CONTEXT WINDOWS ===\n\n")
            for timestamp, context in context_windows.items():
                f.write(f"🕒 EMOTION TIMESTAMP: {timestamp}\n")
                f.write(f"📝 CONTEXT ({context_length}s window):\n")
                f.write(context)
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"💾 Context windows saved to: {context_file}")
        return context_windows
        
    except Exception as e:
        print(f"❌ Error getting transcript context: {e}")
        return {}


def trigger_llm_call(context_window, participant_problems, name):
    #this method will be called when there is a persistent emotion/posture problem 
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    prompt = "You are a helpful assistant, and your job is to respond in a natural, conversational way that fits the situation. If the issue is an emotion, offer encouragement and positivity when it’s a good feeling, or be supportive and constructive if it’s a negative one, look at the meeting context and try to understand what went wrong and offer help. If the issue is posture, gently encourage focus when the person seems disengaged, or reinforce their interest and energy when they appear engaged. You can also offer advice to fix the person’s posture. Keep your response short (1–3 sentences), sound like a teammate rather than an AI, and avoid overused or AI words and phrases. \n\n"
    if context_window != "":
        prompt += "Here is the last two minutes of meeting transcript/context: " + context_window + "\n\n"

    prompt += "Here is the problem that " + name + " is having: " + participant_problems + "\n\n"

    response = client.models.generate_content(
    model="gemini-2.5-flash", contents = prompt)

    return response.text

def send_zoom_message(message, bot_id, person_id = None):

    
    bot_id = f"{bot_id}"  # replace with the actual bot ID
    token = os.getenv('RECALL_API_KEY', '11e4159cb943ded9085f5a8f1ab691c21af45d67') 

    url = f"https://us-west-2.recall.ai/api/v1/bot/%7Bbot_id%7D/send_chat_message/"

    headers = {
        'Authorization': token,
        'accept': 'application/json',
        'content-type': 'application/json'
    }
    if person_id != None:
        data = {
            "to": person_id,
            "message": message
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()
    
    data = {
        "message": message
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

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
