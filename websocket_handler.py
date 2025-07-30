"""
Real-time WebSocket handler for Recall.ai bot connections
Integrates with realtime_ml_app.py for frame and transcript processing
"""

import websocket
import json
import threading
import time
from realtime_ml_app import handle_recall_bot_message, broadcast_to_ui_clients

class RecallBotWebSocketHandler:
    def __init__(self, port=3456):
        self.port = port
        self.server_thread = None
    
    def start_websocket_server(self):
        """Start WebSocket server to receive Recall.ai bot data"""
        def run_server():
            import socket
            import struct
            import hashlib
            import base64
            
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', self.port))
            server_socket.listen(5)
            
            broadcast_to_ui_clients(f"🔌 WebSocket server listening on port {self.port}")
            
            while True:
                try:
                    client_socket, address = server_socket.accept()
                    broadcast_to_ui_clients(f"🤖 Recall.ai bot connected from {address}")
                    
                    # Handle WebSocket handshake
                    self.handle_websocket_handshake(client_socket)
                    
                    # Handle incoming messages
                    threading.Thread(
                        target=self.handle_client_messages, 
                        args=(client_socket,), 
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    broadcast_to_ui_clients(f"❌ WebSocket server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def handle_websocket_handshake(self, client_socket):
        """Handle WebSocket handshake protocol"""
        try:
            request = client_socket.recv(1024).decode('utf-8')
            
            # Extract WebSocket key
            key = None
            for line in request.split('\r\n'):
                if line.startswith('Sec-WebSocket-Key:'):
                    key = line.split(': ')[1].strip()
                    break
            
            if key:
                # Generate accept key
                accept_key = base64.b64encode(
                    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
                ).decode()
                
                # Send handshake response
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
                )
                client_socket.send(response.encode())
                
        except Exception as e:
            broadcast_to_ui_clients(f"❌ WebSocket handshake error: {e}")
    
    def handle_client_messages(self, client_socket):
        """Handle incoming WebSocket messages from Recall.ai bot"""
        try:
            while True:
                # Receive WebSocket frame
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Parse WebSocket frame (simplified)
                if len(data) > 2:
                    # Extract payload (this is a simplified parser)
                    payload_length = data[1] & 127
                    
                    if payload_length < 126:
                        mask_start = 2
                    elif payload_length == 126:
                        mask_start = 4
                    else:
                        mask_start = 10
                    
                    # Extract mask and payload
                    mask = data[mask_start:mask_start + 4]
                    payload = data[mask_start + 4:]
                    
                    # Unmask payload
                    decoded = bytearray()
                    for i, byte in enumerate(payload):
                        decoded.append(byte ^ mask[i % 4])
                    
                    # Process the message
                    try:
                        message = json.loads(decoded.decode('utf-8'))
                        handle_recall_bot_message(message)
                    except json.JSONDecodeError:
                        # Handle binary data (like PNG frames)
                        broadcast_to_ui_clients("📦 Received binary data from bot")
                
        except Exception as e:
            broadcast_to_ui_clients(f"❌ Error handling bot messages: {e}")
        finally:
            client_socket.close()
            broadcast_to_ui_clients("🔌 Recall.ai bot disconnected")

# Initialize and start WebSocket handler
websocket_handler = RecallBotWebSocketHandler()

if __name__ == "__main__":
    websocket_handler.start_websocket_server()
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down WebSocket handler...")
