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
    """Placeholder for ML emotion detection - integrate your model here"""
    
    def __init__(self):
        # Initialize your ML model here
        # self.model = load_your_emotion_model()
        pass
    
    def detect_emotions(self, image_data, participant_metadata):
        """
        Detect emotions from image data (no file saving needed)
        
        Args:
            image_data: numpy array of image (H, W, 3) - direct from PIL/OpenCV
            participant_metadata: dict with participant info and timestamp
        
        Returns:
            dict: ML analysis results
        """
        try:
            # =============================================================================
            # 🔥 NUMPY ARRAY PROCESSING - YOUR ML TEAM GETS CLEAN DATA HERE! 🔥
            # =============================================================================
            
            print(f"\n{'='*80}")
            print(f"🎯 PROCESSING LIVE NUMPY ARRAY FOR ML ANALYSIS")
            print(f"👤 Participant: {participant_metadata.get('participant_name', 'Unknown')}")
            print(f"📊 Image Shape: {image_data.shape}")  # e.g., (480, 640, 3)
            print(f"📈 Data Type: {image_data.dtype}")     # uint8
            print(f"💾 Memory Size: {image_data.nbytes} bytes")
            print(f"🎪 Min/Max Values: {image_data.min()}/{image_data.max()}")
            print(f"📱 Screenshot Type: {participant_metadata.get('screenshot_type', 'unknown')}")
            print(f"⏰ Timestamp: {participant_metadata.get('timestamp', {}).get('absolute', 'N/A')}")
            
            # Show array sample (first few pixels)
            if image_data.size > 0:
                print(f"🔍 First 3 pixels (RGB values):")
                for i in range(min(3, image_data.shape[0])):
                    for j in range(min(3, image_data.shape[1])):
                        rgb = image_data[i, j]
                        print(f"   Pixel [{i},{j}]: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}")
            
            print(f"{'='*80}\n")
            
            # YOUR ML TEAM'S MODELS GO HERE:
            # emotions = your_emotion_model.predict(image_data)
            # productivity = your_productivity_model.predict(image_data)
            # face_detection = your_face_detector.detect(image_data)
            
            # Placeholder emotion detection (replace with actual ML models)
            emotions = {
                'happy': 0.7,
                'focused': 0.8,
                'engaged': 0.6,
                'distracted': 0.2,
                'stressed': 0.1,
                'productivity_score': 0.75,  # Overall productivity metric
                'numpy_array_processed': True,  # Confirmation flag
                'frame_dimensions': f"{image_data.shape[1]}x{image_data.shape[0]}",  # Width x Height
                'total_pixels': image_data.shape[0] * image_data.shape[1]
            }
            
            return emotions
        except Exception as e:
            print(f"❌ Error in emotion detection: {e}")
            return None
    
    def analyze_transcript_sentiment(self, transcript_text):
        """
        Analyze sentiment from transcript
        Replace with your NLP model
        """
        # Placeholder sentiment analysis
        sentiment = {
            'engagement_level': 0.8,
            'focus_score': 0.75,
            'participation_score': 0.7
        }
        return sentiment

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
        
        # Optional: Save file for debugging/backup (DISABLED to prevent storage clogging)
        filename = None
        if False:  # Set to True only if you need to debug/save files
            participant_name = participant.get('name') or f"participant_{participant.get('id')}"
            safe_participant_name = "".join(c for c in participant_name if c.isalnum() or c in ('_', '-'))
            timestamp_str = datetime.fromisoformat(timestamp['absolute'].replace('Z', '+00:00')).strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{safe_participant_name}_{screenshot_type}_{timestamp_str}_{timestamp['relative']}.png"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            with open(filepath, 'wb') as f:
                f.write(buffer)
        
        saved_msg = f"🎯 PROCESSED NUMPY ARRAY: {participant_metadata['participant_name']} ({len(buffer)} bytes) - NO FILE SAVED"
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
    """Save and analyze transcript data"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"transcript_{timestamp}.json"
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        
        # Save transcript
        with open(filepath, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        # Analyze sentiment if transcript text is available
        transcript_text = ""
        if 'data' in transcript_data and 'words' in transcript_data['data']:
            transcript_text = " ".join([word.get('text', '') for word in transcript_data['data']['words']])
        
        sentiment = emotion_detector.analyze_transcript_sentiment(transcript_text)
        
        saved_msg = f"Saved transcript: {filename}"
        if sentiment:
            saved_msg += f" | Engagement: {sentiment.get('engagement_level', 0):.2f}"
        
        broadcast_to_ui_clients(saved_msg, {
            'transcript': transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
            'sentiment': sentiment,
            'filename': filename
        })
        
        return filepath, sentiment
    except Exception as error:
        error_msg = f"Error saving transcript: {str(error)}"
        print(error_msg)
        broadcast_to_ui_clients(error_msg)
        return None, None

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
        # Analyze recent screenshots and transcripts
        stats = {
            'average_productivity': 0.75,
            'total_participants': 3,
            'engagement_level': 0.8,
            'focus_score': 0.7,
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(stats)
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
def send_bot():
    """Handle requests to send a bot to a meeting via Recall.ai API"""
    data = request.get_json()
    meeting_url = data.get('meeting_url')
    bot_name = data.get('bot_name')
    recording_config = data.get('recording_config')
    
    if not meeting_url:
        broadcast_to_ui_clients("Error in /send-bot: meeting_url is required")
        return jsonify({'error': 'meeting_url is required'}), 400
    
    if not RECALL_API_KEY:
        broadcast_to_ui_clients("Error in /send-bot: RECALL_API_KEY is not set in server environment")
        return jsonify({'error': 'RECALL_API_KEY is not set in environment variables'}), 500
    
    try:
        recall_api_url = "https://us-west-2.recall.ai/api/v1/bot"
        
        # Prepare the payload for the Recall.ai API with proper WebSocket URL
        payload = {
            'meeting_url': meeting_url,
            'bot_name': bot_name or 'Productivity Bot',
            'recording_config': {
                'realtime_endpoints': [
                    {
                        'url': 'wss://8083fc32e864.ngrok-free.app',
                        'events': [
                            'video_separate_png.data',
                            'transcript.data',
                            'transcript.partial_data'
                        ]
                    }
                ]
            }
        }
        
        # Override with custom recording_config if provided
        if recording_config:
            payload['recording_config'] = recording_config
        
        broadcast_to_ui_clients("Sending request to Recall.ai API (/v1/bot) with payload:", payload)
        
        response = requests.post(
            recall_api_url,
            json=payload,
            headers={
                'Authorization': f'Token {RECALL_API_KEY}',
                'Content-Type': 'application/json'
            }
        )
        
        if response.status_code == 200 or response.status_code == 201:
            broadcast_to_ui_clients("Successfully called Recall.ai API. Response:", response.json())
            return jsonify(response.json()), response.status_code
        else:
            error_msg = response.json() if response.content else response.text
            broadcast_to_ui_clients("Error calling Recall.ai API:", error_msg)
            return jsonify(error_msg), response.status_code
            
    except Exception as error:
        error_msg = str(error)
        print(f"Error calling Recall.ai API: {error_msg}")
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
            transcript_msg = f"Processing transcript for engagement analysis: {event}"
            filepath, sentiment = save_transcript(message_data['data'])
            broadcast_to_ui_clients(transcript_msg, {
                'sentiment': sentiment,
                'event_type': event
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
                                
                                # Try to parse as JSON
                                try:
                                    json_data = json.loads(message)
                                    event_type = json_data.get('event', 'unknown')
                                    print(f"🎯 Processing event: {event_type}")
                                    handle_recall_bot_message(json_data)
                                except json.JSONDecodeError:
                                    print(f"📝 Non-JSON text message: {message[:100]}...")
                                    
                            except UnicodeDecodeError:
                                print(f"📨 Binary text frame: {len(payload)} bytes")
                        
                        # Process binary frames (opcode 2)
                        elif opcode == 2:
                            print(f"📨 Binary WebSocket frame from {addr}: {len(payload)} bytes")
                            try:
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
                                
                                # Method 3: Try raw binary as UTF-8
                                if not success:
                                    try:
                                        message = payload.decode('utf-8')
                                        print(f"📝 Raw binary as text: {len(message)} chars")
                                        json_data = json.loads(message)
                                        event_type = json_data.get('event', 'unknown')
                                        print(f"🎯 Processing raw binary event: {event_type}")
                                        handle_recall_bot_message(json_data)
                                        success = True
                                    except Exception as e3:
                                        print(f"⚠️ Raw UTF-8 failed: {e3}")
                                
                                # Method 4: Try raw binary as latin-1 (fallback)
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
                                    except Exception as e4:
                                        print(f"⚠️ Latin-1 failed: {e4}")
                                
                                if not success:
                                    print(f"❓ All decoding methods failed. Raw bytes sample: {payload[:50]}")
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
    print(f"🚀 Flask Productivity Tracker running at http://localhost:{PORT}")
    print(f"📊 ML emotion detection ready for video analysis")
    print(f"📝 Transcript sentiment analysis enabled")
    print(f"💾 Screenshots saved to: {SCREENSHOTS_DIR}")
    print(f"📋 Transcripts saved to: {TRANSCRIPTS_DIR}")
    print(f"🔗 Connect Recall.ai bot WebSocket to: wss://8083fc32e864.ngrok-free.app")
    print(f"🌐 NGROK URL for bot webhook: https://8083fc32e864.ngrok-free.app")
    
    # Start WebSocket server for Recall.ai bot
    start_websocket_server()
    
    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
