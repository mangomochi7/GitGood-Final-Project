from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import requests
import base64
import os
import json
from datetime import datetime
import threading
import time
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import io
import socket
import struct
import hashlib

# Import emotion detection functionality
try:
    from emotion_by_images import get_emotion
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace emotion detection loaded successfully")
except ImportError as e:
    print(f"⚠️  DeepFace not available: {e}")
    print("🔄 Using placeholder emotion detection")
    DEEPFACE_AVAILABLE = False
    
    def get_emotion(frame):
        """Placeholder emotion detection when DeepFace is not available"""
        import random
        # Return random emotions for testing
        emotions = ['happy', 'neutral', 'surprise', 'sad', 'angry', 'fear', 'disgust']
        return [random.choice(emotions)]

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
RECALL_API_KEY = os.getenv('RECALL_API_KEY')
PORT = int(os.getenv('PORT', 5000))
WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', 3456))

# Create directories
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed_frames')
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), 'transcripts')

for directory in [SCREENSHOTS_DIR, PROCESSED_DIR, TRANSCRIPTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

class EmotionDetector:
    """Real-time emotion detection using DeepFace model"""
    
    def __init__(self):
        """Initialize the emotion detector with participant tracking"""
        print("🧠 Initializing DeepFace emotion detection model...")
        self.participant_emotions = {}  # Track emotions per participant
        self.emotion_history = {}       # Track emotion history per participant
        self.frame_counts = {}          # Track frame counts per participant
        
        # Test DeepFace availability
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print("✅ DeepFace model loaded successfully")
        except ImportError as e:
            print(f"❌ DeepFace not available: {e}")
            self.deepface = None
    
    def detect_emotions(self, image_data, participant_metadata):
        """
        Detect emotions from image data using DeepFace
        
        Args:
            image_data: numpy array of image (H, W, 3) - RGB format from PIL
            participant_metadata: dict with participant info and timestamp
        
        Returns:
            dict: ML analysis results with emotions and productivity metrics
        """
        try:
            participant_id = participant_metadata.get('participant_id', 'unknown')
            participant_name = participant_metadata.get('participant_name', f'participant_{participant_id}')
            
            print(f"\n{'='*80}")
            print(f"🎯 PROCESSING PARTICIPANT: {participant_name} (ID: {participant_id})")
            print(f"📊 Image Shape: {image_data.shape}")
            print(f"� Data Type: {image_data.dtype}")
            print(f"💾 Memory Size: {image_data.nbytes} bytes")
            print(f"📱 Screenshot Type: {participant_metadata.get('screenshot_type', 'unknown')}")
            print(f"⏰ Timestamp: {participant_metadata.get('timestamp', {}).get('absolute', 'N/A')}")
            print(f"{'='*80}")
            
            # Initialize participant tracking if new
            if participant_id not in self.participant_emotions:
                self.participant_emotions[participant_id] = {
                    'name': participant_name,
                    'latest_emotions': {},
                    'dominant_emotion': 'neutral',
                    'confidence': 0.0
                }
                self.emotion_history[participant_id] = []
                self.frame_counts[participant_id] = 0
            
            self.frame_counts[participant_id] += 1
            
            # Convert PIL RGB to OpenCV BGR format for emotion detection
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # Convert RGB to BGR for OpenCV compatibility
                bgr_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            else:
                print(f"⚠️ Unexpected image format: {image_data.shape}")
                bgr_image = image_data
            
            # Use the emotion detection model
            if self.deepface is not None:
                detected_emotions = get_emotion(bgr_image)
                print(f"🎭 Detected emotions for {participant_name}: {detected_emotions}")
                
                # Process detected emotions
                if detected_emotions and len(detected_emotions) > 0:
                    # For multiple faces, use the first detected emotion
                    primary_emotion = detected_emotions[0] if detected_emotions else 'neutral'
                    
                    # Update participant's emotion state
                    self.participant_emotions[participant_id]['dominant_emotion'] = primary_emotion
                    self.participant_emotions[participant_id]['latest_emotions'] = {
                        'primary': primary_emotion,
                        'all_faces': detected_emotions,
                        'face_count': len(detected_emotions)
                    }
                    
                    # Add to emotion history (keep last 10 entries)
                    self.emotion_history[participant_id].append({
                        'emotion': primary_emotion,
                        'timestamp': participant_metadata.get('timestamp', {}),
                        'face_count': len(detected_emotions)
                    })
                    
                    if len(self.emotion_history[participant_id]) > 10:
                        self.emotion_history[participant_id] = self.emotion_history[participant_id][-10:]
                    
                    # Calculate productivity score based on emotions
                    productivity_score = self._calculate_productivity_score(primary_emotion, detected_emotions)
                    
                else:
                    print(f"😶 No faces detected for {participant_name}")
                    primary_emotion = 'no_face_detected'
                    productivity_score = 0.3  # Lower score when no face is detected
                
            else:
                print("❌ DeepFace not available, using placeholder")
                primary_emotion = 'unknown'
                detected_emotions = []
                productivity_score = 0.5
            
            # Compile results
            emotion_results = {
                'participant_id': participant_id,
                'participant_name': participant_name,
                'dominant_emotion': primary_emotion,
                'all_emotions': detected_emotions,
                'face_count': len(detected_emotions) if detected_emotions else 0,
                'productivity_score': productivity_score,
                'engagement_level': self._calculate_engagement_level(primary_emotion),
                'frame_number': self.frame_counts[participant_id],
                'processing_successful': True,
                'model_used': 'DeepFace' if self.deepface else 'Placeholder',
                'numpy_array_processed': True,
                'frame_dimensions': f"{image_data.shape[1]}x{image_data.shape[0]}",
                'total_pixels': image_data.shape[0] * image_data.shape[1],
                'timestamp': participant_metadata.get('timestamp', {})
            }
            
            # Add emotion history summary
            if participant_id in self.emotion_history and self.emotion_history[participant_id]:
                recent_emotions = [entry['emotion'] for entry in self.emotion_history[participant_id][-5:]]
                emotion_results['recent_emotion_trend'] = recent_emotions
                emotion_results['emotion_consistency'] = len(set(recent_emotions)) <= 2  # True if consistent
            
            print(f"✅ Emotion analysis complete for {participant_name}")
            print(f"   🎭 Primary emotion: {primary_emotion}")
            print(f"   📊 Productivity score: {productivity_score:.2f}")
            print(f"   👥 Faces detected: {len(detected_emotions) if detected_emotions else 0}")
            print(f"   🖼️  Frame #{self.frame_counts[participant_id]} processed")
            
            # Clear terminal output for easy reading
            print(f"\n🎯 EMOTION RESULT: {participant_name} → {primary_emotion.upper()}")
            if detected_emotions and len(detected_emotions) > 0:
                print(f"   ✅ Face detected - Emotion: {primary_emotion}")
                print(f"   📈 Productivity: {productivity_score:.1%}")
                print(f"   💪 Engagement: {self._calculate_engagement_level(primary_emotion):.1%}")
            else:
                print(f"   ❌ No face detected in video frame")
            print("-" * 60)
            
            return emotion_results
            
        except Exception as e:
            print(f"❌ Error in emotion detection for participant {participant_metadata.get('participant_name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'participant_id': participant_metadata.get('participant_id', 'unknown'),
                'participant_name': participant_metadata.get('participant_name', 'unknown'),
                'error': str(e),
                'processing_successful': False,
                'dominant_emotion': 'error',
                'productivity_score': 0.0
            }
    
    def _calculate_productivity_score(self, primary_emotion, all_emotions):
        """Calculate productivity score based on detected emotions"""
        emotion_productivity_map = {
            'happy': 0.8,
            'neutral': 0.7,
            'surprise': 0.6,
            'sad': 0.4,
            'fear': 0.3,
            'angry': 0.2,
            'disgust': 0.3,
            'no_face_detected': 0.3,
            'error': 0.0,
            'unknown': 0.5
        }
        
        base_score = emotion_productivity_map.get(primary_emotion, 0.5)
        
        # Bonus for engagement (having face detected)
        if all_emotions and len(all_emotions) > 0:
            base_score += 0.1
        
        # Multiple people might indicate collaboration
        if all_emotions and len(all_emotions) > 1:
            base_score += 0.05
        
        return min(1.0, max(0.0, base_score))
    
    def _calculate_engagement_level(self, primary_emotion):
        """Calculate engagement level based on emotion"""
        engagement_map = {
            'happy': 0.9,
            'surprise': 0.8,
            'neutral': 0.6,
            'sad': 0.4,
            'angry': 0.5,
            'fear': 0.3,
            'disgust': 0.3,
            'no_face_detected': 0.2,
            'error': 0.0,
            'unknown': 0.5
        }
        return engagement_map.get(primary_emotion, 0.5)
    
    def get_all_participants_summary(self):
        """Get summary of all participants' current emotional states"""
        summary = {
            'total_participants': len(self.participant_emotions),
            'participants': [],
            'overall_productivity': 0.0,
            'overall_engagement': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        total_productivity = 0
        total_engagement = 0
        active_participants = 0
        
        for participant_id, data in self.participant_emotions.items():
            participant_summary = {
                'id': participant_id,
                'name': data['name'],
                'current_emotion': data['dominant_emotion'],
                'frame_count': self.frame_counts.get(participant_id, 0),
                'recent_emotions': [entry['emotion'] for entry in self.emotion_history.get(participant_id, [])[-3:]]
            }
            
            # Calculate individual scores
            if data['dominant_emotion'] not in ['error', 'unknown', 'no_face_detected']:
                participant_productivity = self._calculate_productivity_score(data['dominant_emotion'], [])
                participant_engagement = self._calculate_engagement_level(data['dominant_emotion'])
                
                participant_summary['productivity_score'] = participant_productivity
                participant_summary['engagement_level'] = participant_engagement
                
                total_productivity += participant_productivity
                total_engagement += participant_engagement
                active_participants += 1
            
            summary['participants'].append(participant_summary)
        
        # Calculate overall scores
        if active_participants > 0:
            summary['overall_productivity'] = total_productivity / active_participants
            summary['overall_engagement'] = total_engagement / active_participants
        
        return summary

# Initialize emotion detector
emotion_detector = EmotionDetector()

def save_png_screenshot(base64_data, participant, timestamp, screenshot_type):
    """Process PNG frame and send clean data to ML team"""
    try:
        # Remove data URL prefix if present
        base64_clean = base64_data.replace('data:image/png;base64,', '')
        
        # Convert base64 to bytes
        buffer = base64.b64decode(base64_clean)
        
        # Convert to PIL Image (clean PNG)
        image = Image.open(io.BytesIO(buffer))
        
        # Convert to numpy array for ML processing
        image_array = np.array(image)
        
        # Prepare metadata for ML team
        participant_metadata = {
            'participant_id': participant.get('id'),
            'participant_name': participant.get('name') or f"participant_{participant.get('id')}",
            'timestamp': timestamp,
            'frame_size': image.size,  # (width, height)
            'image_shape': image_array.shape,  # (height, width, channels)
            'screenshot_type': screenshot_type,  # 'webcam' or 'screenshare'
            'format': 'PNG'
        }
        
        # SEND CLEAN DATA TO ML TEAM (no file saving needed)
        emotions = emotion_detector.detect_emotions(image_array, participant_metadata)
        
        # TEMPORARILY ENABLED: Save file for debugging to see if images are coming through
        filename = None
        if True:  # ENABLED to test if images are being recorded
            participant_name = participant.get('name') or f"participant_{participant.get('id')}"
            safe_participant_name = "".join(c for c in participant_name if c.isalnum() or c in ('_', '-'))
            timestamp_str = datetime.fromisoformat(timestamp['absolute'].replace('Z', '+00:00')).strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{safe_participant_name}_{screenshot_type}_{timestamp_str}_{timestamp['relative']}.png"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            with open(filepath, 'wb') as f:
                f.write(buffer)
            print(f"💾 SAVED IMAGE FILE: {filename} ({len(buffer)} bytes)")
        
        saved_msg = f"🎯 PROCESSED NUMPY ARRAY: {participant_metadata['participant_name']} ({len(buffer)} bytes)"
        if filename:
            saved_msg += f" | SAVED: {filename}"
        else:
            saved_msg += " | NO FILE SAVED"
        if emotions:
            saved_msg += f" | Productivity: {emotions.get('productivity_score', 0):.2f}"
            saved_msg += f" | Pixels: {emotions.get('total_pixels', 0):,}"
        
        print(saved_msg)
        broadcast_to_ui_clients(saved_msg, {
            'participant': participant,
            'emotions': emotions,
            'filename': filename,
            'metadata': participant_metadata
        })
        
        return image_array, emotions  # Return clean data instead of file path
    except Exception as error:
        error_msg = f"Error processing PNG frame: {str(error)}"
        print(error_msg)
        broadcast_to_ui_clients(error_msg)
        return None, None

def save_transcript(transcript_data):
    """Save transcript data as plain text file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"transcript_{timestamp}.txt"
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        
        # Extract transcript text
        transcript_text = ""
        if 'data' in transcript_data and 'words' in transcript_data['data']:
            transcript_text = " ".join([word.get('text', '') for word in transcript_data['data']['words']])
        elif isinstance(transcript_data, dict) and 'text' in transcript_data:
            transcript_text = transcript_data['text']
        elif isinstance(transcript_data, str):
            transcript_text = transcript_data
        else:
            transcript_text = str(transcript_data)
        
        # Save transcript as plain text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(transcript_text)
            f.write("\n")
        
        saved_msg = f"📝 Saved transcript: {filename} ({len(transcript_text)} chars)"
        print(saved_msg)
        broadcast_to_ui_clients(saved_msg, {
            'transcript': transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
            'filename': filename,
            'char_count': len(transcript_text)
        })
        
        return filepath
    except Exception as error:
        error_msg = f"Error saving transcript: {str(error)}"
        print(error_msg)
        broadcast_to_ui_clients(error_msg)
        return None

def broadcast_to_ui_clients(log_message, data=None):
    """Send a message to all connected UI clients"""
    message = {
        'log': log_message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    socketio.emit('ui_update', message)
    print(f"[UI Broadcast] {log_message}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/productivity-stats')
def get_productivity_stats():
    """Get productivity statistics from processed data"""
    try:
        # Get real-time participant summary from emotion detector
        participant_summary = emotion_detector.get_all_participants_summary()
        
        # Combine with any additional stats
        stats = {
            'average_productivity': participant_summary.get('overall_productivity', 0.75),
            'total_participants': participant_summary.get('total_participants', 0),
            'engagement_level': participant_summary.get('overall_engagement', 0.8),
            'last_updated': datetime.now().isoformat(),
            'participants': participant_summary.get('participants', []),
            'detailed_summary': participant_summary
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/participants')
def get_participants():
    """Get current participant list with their emotional states"""
    try:
        summary = emotion_detector.get_all_participants_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/participant/<participant_id>')
def get_participant_details(participant_id):
    """Get detailed information about a specific participant"""
    try:
        if participant_id in emotion_detector.participant_emotions:
            participant_data = emotion_detector.participant_emotions[participant_id]
            emotion_history = emotion_detector.emotion_history.get(participant_id, [])
            frame_count = emotion_detector.frame_counts.get(participant_id, 0)
            
            details = {
                'participant_id': participant_id,
                'name': participant_data['name'],
                'current_state': participant_data,
                'emotion_history': emotion_history,
                'total_frames_processed': frame_count,
                'last_updated': datetime.now().isoformat()
            }
            return jsonify(details)
        else:
            return jsonify({'error': f'Participant {participant_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/screenshots')
def list_screenshots():
    """List all saved screenshots"""
    try:
        files = os.listdir(SCREENSHOTS_DIR)
        screenshots = [f for f in files if f.endswith('.png')]
        return jsonify({'screenshots': screenshots})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    """Serve screenshot files"""
    return send_from_directory(SCREENSHOTS_DIR, filename)

@app.route('/test-numpy-array', methods=['POST'])
def test_numpy_array():
    """Test endpoint to demonstrate numpy array processing with a sample PNG"""
    try:
        print("🧪 Creating test numpy array and PNG for demonstration...")
        
        # Create a test image array (100x100 RGB with gradient pattern)
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a gradient pattern for testing
        for i in range(100):
            for j in range(100):
                test_array[i, j] = [i * 2, j * 2, (i + j) % 255]
        
        # Convert numpy array to PIL Image
        test_image = Image.fromarray(test_array)
        
        # Convert to base64 PNG (simulating what Recall.ai would send)
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create fake participant data
        fake_participant = {
            'id': 'test_participant_123',
            'name': 'Test User'
        }
        
        fake_timestamp = {
            'absolute': '2025-07-30T14:48:00Z',
            'relative': 1000
        }
        
        print("🎯 Testing numpy array processing pipeline...")
        
        # Process the PNG through our ML pipeline
        processed_array, emotions = save_png_screenshot(
            base64_data,
            fake_participant,
            fake_timestamp,
            'webcam'
        )
        
        if processed_array is not None:
            return jsonify({
                'success': True,
                'message': 'Numpy array processing test successful!',
                'array_shape': processed_array.shape,
                'array_dtype': str(processed_array.dtype),
                'array_size_bytes': processed_array.nbytes,
                'emotions': emotions,
                'sample_pixels': {
                    'top_left': processed_array[0, 0].tolist(),
                    'center': processed_array[50, 50].tolist(),
                    'bottom_right': processed_array[99, 99].tolist()
                }
            })
        else:
            return jsonify({'error': 'Failed to process array'}), 500
            
    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/send-bot', methods=['POST'])
@app.route('/api/send-bot', methods=['POST'])
def send_bot():
    """Handle requests to send a bot to a meeting via Recall.ai API"""
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
        broadcast_to_ui_clients(error_msg)
        return jsonify({'error': 'meeting_url is required'}), 400
    
    if not RECALL_API_KEY:
        error_msg = "Error in /send-bot: RECALL_API_KEY is not set in server environment"
        print(f"❌ {error_msg}")
        broadcast_to_ui_clients(error_msg)
        return jsonify({'error': 'RECALL_API_KEY is not set in environment variables'}), 500
    
    try:
        recall_api_url = "https://us-west-2.recall.ai/api/v1/bot"
        
        # Build payload - use provided recording_config or fallback to default
        payload = {
            'meeting_url': meeting_url,
            'bot_name': bot_name or 'Productivity Bot'
        }
        
        # If recording_config is provided by frontend, use it
        if recording_config:
            payload['recording_config'] = recording_config
            print(f"📋 Using frontend recording config with events: {recording_config.get('realtime_endpoints', [{}])[0].get('events', [])}")
        else:
            # Fallback to default config for video and transcript
            payload['recording_config'] = {
                'realtime_endpoints': [
                    {
                        'url': 'wss://60fc0a664840.ngrok-free.app',
                        'events': [
                            'video_separate_png.data',
                            'transcript.data',
                            'transcript.partial_data'
                        ]
                    }
                ]
            }
            print(f"📋 Using default recording config")
        
        print(f"📤 Sending request to Recall.ai API...")
        broadcast_to_ui_clients("Sending request to Recall.ai API (/v1/bot) with payload:", payload)
        
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
            broadcast_to_ui_clients(success_msg, response.json())
            return jsonify(response.json()), response.status_code
        else:
            error_msg = response.json() if response.content else response.text
            print(f"❌ Recall.ai API Error: {response.status_code} - {error_msg}")
            broadcast_to_ui_clients("Error calling Recall.ai API:", error_msg)
            return jsonify(error_msg), response.status_code
            
    except Exception as error:
        error_msg = str(error)
        print(f"❌ Error calling Recall.ai API: {error_msg}")
        broadcast_to_ui_clients("Error calling Recall.ai API:", error_msg)
        return jsonify({'error': 'Failed to send bot to meeting due to an internal server error.'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle UI WebSocket connection"""
    print("UI WebSocket client connected")
    emit('ui_update', {
        'log': 'Connected to Flask Productivity Tracker.',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle UI WebSocket disconnection"""
    print("UI WebSocket client disconnected")

# Simulated WebSocket message handler for Recall.ai bot
def handle_recall_bot_message(message_data):
    """Handle messages from Recall.ai bot - integrate with your WebSocket server"""
    try:
        event = message_data.get('event')
        
        if event == "video_separate_png.data":
            participant_info = message_data['data']['data']['participant']
            video_msg = f"Processing video for productivity analysis: {participant_info.get('name') or participant_info.get('id')}"
            
            # Save and process the screenshot
            filepath, emotions = save_png_screenshot(
                message_data['data']['data']['buffer'],
                participant_info,
                message_data['data']['data']['timestamp'],
                message_data['data']['data']['type']
            )
            
            broadcast_to_ui_clients(video_msg, {
                'participant': participant_info,
                'emotions': emotions,
                'type': message_data['data']['data']['type']
            })
            
        elif event in ["transcript.data", "transcript.partial_data"]:
            transcript_msg = f"📝 Processing transcript: {event}"
            print(transcript_msg)
            filepath = save_transcript(message_data['data'])
            broadcast_to_ui_clients(transcript_msg, {
                'event_type': event,
                'saved_to': filepath
            })
            
    except Exception as e:
        error_msg = f"Error processing Recall.ai message: {str(e)}"
        print(error_msg)
        broadcast_to_ui_clients(error_msg)

def start_websocket_server():
    """Start a proper WebSocket server for Recall.ai bot data"""
    
    def handle_websocket_client(client_socket, addr):
        """Handle a WebSocket client connection"""
        try:
            print(f"🤝 Handling WebSocket connection from {addr}")
            
            # Read the HTTP request
            request_data = b""
            while b"\r\n\r\n" not in request_data:
                chunk = client_socket.recv(1024)
                if not chunk:
                    print(f"❌ No data received from {addr}")
                    return
                request_data += chunk
            
            request = request_data.decode('utf-8')
            print(f"🔍 WebSocket request received from {addr}")
            print(f"📋 Request headers:\n{request}")
            
            # Parse headers
            lines = request.split('\r\n')
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Check if it's a WebSocket upgrade request
            if (headers.get('upgrade', '').lower() != 'websocket' or 
                headers.get('connection', '').lower() != 'upgrade'):
                print(f"❌ Not a WebSocket upgrade request from {addr}")
                client_socket.close()
                return
            
            # Get WebSocket key
            websocket_key = headers.get('sec-websocket-key')
            if not websocket_key:
                print(f"❌ No Sec-WebSocket-Key header from {addr}")
                client_socket.close()
                return
            
            print(f"🔑 WebSocket key: {websocket_key}")
            
            # Generate accept key
            import base64
            accept_key = base64.b64encode(
                hashlib.sha1((websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            ).decode()
            
            # Send WebSocket handshake response
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n"
                "\r\n"
            )
            
            print(f"📤 Sending WebSocket handshake response to {addr}")
            client_socket.send(response.encode())
            print(f"✅ WebSocket handshake successful with {addr}")
            
            # Now handle WebSocket frames
            while True:
                try:
                    # Read WebSocket frame
                    frame_data = client_socket.recv(4096)
                    if not frame_data:
                        print(f"🔌 WebSocket connection closed by {addr}")
                        break
                    
                    if len(frame_data) < 2:
                        continue
                    
                    # Parse WebSocket frame (basic implementation)
                    opcode = frame_data[0] & 0x0F
                    masked = (frame_data[1] & 0x80) == 0x80
                    payload_length = frame_data[1] & 0x7F
                    
                    header_length = 2
                    if payload_length == 126:
                        payload_length = struct.unpack('>H', frame_data[2:4])[0]
                        header_length = 4
                    elif payload_length == 127:
                        payload_length = struct.unpack('>Q', frame_data[2:10])[0]
                        header_length = 10
                    
                    if masked:
                        mask = frame_data[header_length:header_length + 4]
                        header_length += 4
                    
                    # Extract payload
                    if len(frame_data) >= header_length + payload_length:
                        payload = frame_data[header_length:header_length + payload_length]
                        
                        if masked:
                            # Unmask payload
                            payload = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
                        
                        # Process text frames (opcode 1)
                        if opcode == 1:
                            try:
                                message = payload.decode('utf-8')
                                print(f"📨 WebSocket text message from {addr}: {len(message)} chars")
                                
                                # NEW: Try to clean up corrupted text data
                                # Replace non-printable characters and try to extract JSON
                                import re
                                # Try to find JSON-like structures in corrupted text
                                clean_message = ''.join(char if ord(char) >= 32 and ord(char) < 127 else ' ' for char in message)
                                print(f"🧹 Cleaned message: {clean_message[:100]}...")
                                
                                # Look for JSON patterns
                                json_matches = re.findall(r'\{[^{}]*\}', clean_message)
                                for json_match in json_matches:
                                    try:
                                        json_data = json.loads(json_match)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 Found JSON in text! Processing event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        continue
                                    except:
                                        pass
                                
                                # Try to parse as JSON directly
                                try:
                                    json_data = json.loads(message)
                                    event_type = json_data.get('event', 'unknown')
                                    print(f"🎯 Processing event: {event_type}")
                                    handle_recall_bot_message(json_data)
                                except json.JSONDecodeError:
                                    print(f"📝 Non-JSON text message: {message[:100]}...")
                                    
                                    # NEW: Try different character encodings
                                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                                        try:
                                            # Re-decode with different encoding
                                            alt_message = payload.decode(encoding)
                                            if alt_message.strip().startswith('{') and '"event"' in alt_message:
                                                json_data = json.loads(alt_message)
                                                event_type = json_data.get('event', 'unknown')
                                                print(f"🎯 SUCCESS with {encoding}! Processing event: {event_type}")
                                                handle_recall_bot_message(json_data)
                                                break
                                        except:
                                            continue
                                    
                            except UnicodeDecodeError:
                                print(f"📨 Binary text frame: {len(payload)} bytes")
                                # Try processing as binary data
                                try:
                                    # Check if first byte indicates JSON start
                                    if len(payload) > 0 and payload[0] == 0x7b:  # '{' character
                                        # Try to extract JSON from binary data
                                        json_bytes = bytearray()
                                        brace_count = 0
                                        for byte in payload:
                                            if byte == 0x7b:  # '{'
                                                brace_count += 1
                                            elif byte == 0x7d:  # '}'
                                                brace_count -= 1
                                            
                                            if 32 <= byte <= 126:  # Printable ASCII
                                                json_bytes.append(byte)
                                            else:
                                                json_bytes.append(32)  # Replace with space
                                            
                                            if brace_count == 0 and len(json_bytes) > 0:
                                                break
                                        
                                        try:
                                            potential_json = json_bytes.decode('utf-8').strip()
                                            print(f"🔍 Extracted potential JSON: {potential_json[:100]}...")
                                            json_data = json.loads(potential_json)
                                            event_type = json_data.get('event', 'unknown')
                                            print(f"🎯 SUCCESS! Extracted JSON event: {event_type}")
                                            handle_recall_bot_message(json_data)
                                        except:
                                            print(f"⚠️ Failed to parse extracted JSON")
                                except Exception as e:
                                    print(f"⚠️ Binary text processing failed: {e}")
                        
                        # Process binary frames (opcode 2)
                        elif opcode == 2:
                            print(f"📨 Binary WebSocket frame from {addr}: {len(payload)} bytes")
                            try:
                                # NEW: Enhanced debugging - save binary data and try more methods
                                print(f"🔍 BINARY DATA ANALYSIS:")
                                print(f"   📏 Payload length: {len(payload)} bytes")
                                print(f"   🎯 First 20 bytes: {[hex(b) for b in payload[:20]]}")
                                print(f"   📄 ASCII interpretation: {payload[:50]}")
                                
                                # Check if it's actually text data that got sent as binary
                                try:
                                    # Method 0: Check if it's actually text JSON sent as binary
                                    message = payload.decode('utf-8', errors='ignore')
                                    print(f"📝 UTF-8 decoded (with errors ignored): {message[:100]}...")
                                    
                                    if message.strip().startswith('{') and '"event"' in message:
                                        json_data = json.loads(message)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 SUCCESS! Processing UTF-8 event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        continue
                                except Exception as e0:
                                    print(f"⚠️ UTF-8 with error ignore failed: {e0}")
                                
                                # Try multiple decompression methods
                                import gzip, zlib
                                success = False
                                
                                # Method 1: Try gzip decompression
                                try:
                                    decompressed = gzip.decompress(payload)
                                    message = decompressed.decode('utf-8')
                                    print(f"🗜️ GZIP decompressed: {len(message)} chars")
                                    json_data = json.loads(message)
                                    event_type = json_data.get('event', 'unknown')
                                    print(f"🎯 Processing GZIP event: {event_type}")
                                    handle_recall_bot_message(json_data)
                                    success = True
                                except Exception as e1:
                                    print(f"⚠️ GZIP failed: {e1}")
                                
                                # Method 2: Try zlib decompression
                                if not success:
                                    try:
                                        decompressed = zlib.decompress(payload)
                                        message = decompressed.decode('utf-8')
                                        print(f"🗜️ ZLIB decompressed: {len(message)} chars")
                                        json_data = json.loads(message)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 Processing ZLIB event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        success = True
                                    except Exception as e2:
                                        print(f"⚠️ ZLIB failed: {e2}")
                                
                                # Method 3: Try deflate decompression
                                if not success:
                                    try:
                                        decompressed = zlib.decompress(payload, -zlib.MAX_WBITS)
                                        message = decompressed.decode('utf-8')
                                        print(f"🗜️ DEFLATE decompressed: {len(message)} chars")
                                        json_data = json.loads(message)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 Processing DEFLATE event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        success = True
                                    except Exception as e3:
                                        print(f"⚠️ DEFLATE failed: {e3}")
                                
                                # Method 4: Try raw binary as UTF-8
                                if not success:
                                    try:
                                        message = payload.decode('utf-8')
                                        print(f"📝 Raw binary as text: {len(message)} chars")
                                        json_data = json.loads(message)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 Processing raw binary event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        success = True
                                    except Exception as e4:
                                        print(f"⚠️ Raw UTF-8 failed: {e4}")
                                
                                # Method 5: Try raw binary as latin-1 (fallback)
                                if not success:
                                    try:
                                        message = payload.decode('latin-1')
                                        print(f"📝 Raw binary as latin-1: {len(message)} chars")
                                        # Check if it looks like JSON
                                        if message.strip().startswith('{') and '"event"' in message:
                                            json_data = json.loads(message)
                                            event_type = json_data.get('event', 'unknown')
                                            print(f"🎯 Processing latin-1 event: {event_type}")
                                            handle_recall_bot_message(json_data)
                                            success = True
                                    except Exception as e5:
                                        print(f"⚠️ Latin-1 failed: {e5}")
                                
                                if not success:
                                    print(f"❓ All decoding methods failed. Saving binary data for analysis...")
                                    # Save binary payload to file for analysis
                                    binary_filename = f"debug_binary_{datetime.now().strftime('%H-%M-%S')}.bin"
                                    binary_path = os.path.join(SCREENSHOTS_DIR, binary_filename)
                                    with open(binary_path, 'wb') as f:
                                        f.write(payload)
                                    print(f"💾 Saved binary data to: {binary_filename}")
                                    print(f"🔍 Raw bytes sample: {payload[:50]}")
                                    print(f"🔍 Byte values: {[hex(b) for b in payload[:20]]}")
                                    
                            except Exception as e:
                                print(f"❌ Critical error processing binary frame: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        
                        # Handle close frames (opcode 8)
                        elif opcode == 8:
                            print(f"🔌 WebSocket close frame from {addr}")
                            break
                            
                except Exception as e:
                    print(f"❌ Error processing WebSocket frame from {addr}: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ WebSocket client error from {addr}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                client_socket.close()
                print(f"🔌 WebSocket connection closed: {addr}")
            except:
                pass
    
    def run_server():
        """Run the WebSocket server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind(('0.0.0.0', WEBSOCKET_PORT))
            server.listen(10)
            print(f"🔌 WebSocket server listening on port {WEBSOCKET_PORT}")
            broadcast_to_ui_clients(f"WebSocket server ready on port {WEBSOCKET_PORT}")
            
            while True:
                try:
                    client, addr = server.accept()
                    print(f"🤝 New connection from {addr}")
                    
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=handle_websocket_client, 
                        args=(client, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    print(f"❌ Server accept error: {e}")
                    
        except Exception as e:
            print(f"❌ WebSocket server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            server.close()
    
    # Start the WebSocket server in a separate thread
    websocket_thread = threading.Thread(target=run_server, daemon=True)
    websocket_thread.start()
    
    return websocket_thread

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 ZOOM EMOTION DETECTION FLASK APP")
    print("="*80)
    print(f"🌐 Server: http://localhost:{PORT}")
    print(f"� WebSocket: Port {WEBSOCKET_PORT}")
    print(f"🎭 ML Model: DeepFace emotion detection {'✅ READY' if DEEPFACE_AVAILABLE else '❌ UNAVAILABLE'}")
    print(f"� Transcripts: {TRANSCRIPTS_DIR}")
    print(f"🔗 Recall.ai WebSocket: wss://4e7f581a2f1b.ngrok-free.app")
    print("="*80)
    print("📊 WHAT YOU'LL SEE:")
    print("   • Real-time emotion detection for each Zoom participant")
    print("   • Participant emotions displayed in terminal")
    print("   • Productivity & engagement scores")
    print("   • Transcripts saved as .txt files")
    print("="*80)
    print("🎯 Waiting for Zoom participant video data...")
    print("   Send a bot to meeting → Video frames → Emotion detection!")
    print("="*80 + "\n")
    
    # Start WebSocket server for Recall.ai bot
    start_websocket_server()
    
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
