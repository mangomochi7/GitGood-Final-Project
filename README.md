# GitGood Final Project - Real-time Emotion Detection for Zoom Meetings

🎯 **Real-time emotion detection and productivity analysis for Zoom meeting participants using AI and computer vision.**

## 🚀 Overview

This project integrates with [Recall.ai](https://recall.ai) to capture live video streams from Zoom meeting participants and analyzes their emotions in real-time using DeepFace. The system provides:

- **Real-time emotion detection** for each participant
- **Productivity scoring** based on facial expressions
- **Multi-participant tracking** with individual emotion histories
- **Transcript sentiment analysis** for engagement metrics
- **RESTful API** for accessing participant data
- **WebSocket streaming** for live updates

## 🏗️ Architecture

```
Zoom Meeting → Recall.ai Bot → WebSocket Stream → Flask App → DeepFace → Emotion Analysis
                                     ↓
                              Web Dashboard ← RESTful API ← Participant Tracking
```

## 📦 Installation

### 1. Quick Setup
```bash
# Install all dependencies automatically
python install_dependencies.py
```

### 2. Manual Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Key packages installed:
# - Flask & Flask-SocketIO (web server)
# - DeepFace (emotion detection)
# - OpenCV (computer vision)
# - TensorFlow (ML backend)
# - Recall.ai WebSocket handling
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
RECALL_API_KEY=your_recall_api_key_here
PORT=5000
WEBSOCKET_PORT=3456
```

## 🎭 How Emotion Detection Works

### Multi-Participant Processing
The system handles multiple participants simultaneously:

```python
# Each participant gets tracked individually
participant_emotions = {
    'user_001': {
        'name': 'Alice',
        'dominant_emotion': 'happy',
        'productivity_score': 0.85,
        'frame_count': 42
    },
    'user_002': {
        'name': 'Bob', 
        'dominant_emotion': 'neutral',
        'productivity_score': 0.70,
        'frame_count': 38
    }
}
```

### Emotion Detection Pipeline

1. **Frame Capture**: Recall.ai bot captures individual participant video streams
2. **Image Processing**: Each frame is converted to numpy arrays (H, W, 3)
3. **Face Detection**: OpenCV detects faces in each participant's frame
4. **Emotion Analysis**: DeepFace analyzes detected faces for emotions
5. **Scoring**: Custom algorithms calculate productivity and engagement scores
6. **Tracking**: Individual participant emotion histories are maintained

### Supported Emotions
- **Happy** 😊 (Productivity: 0.8)
- **Neutral** 😐 (Productivity: 0.7) 
- **Surprise** 😲 (Productivity: 0.6)
- **Sad** 😢 (Productivity: 0.4)
- **Angry** 😠 (Productivity: 0.2)
- **Fear** 😨 (Productivity: 0.3)
- **Disgust** 🤢 (Productivity: 0.3)

## 🔧 Usage

### 1. Start the Application
```bash
python app_ml.py
```

The server will start on `http://localhost:5000` with:
- WebSocket server on port 3456 for Recall.ai bot connections
- RESTful API endpoints for data access
- Real-time web dashboard

### 2. Send Bot to Zoom Meeting
```bash
curl -X POST http://localhost:5000/api/send-bot \
  -H "Content-Type: application/json" \
  -d '{
    "meeting_url": "https://zoom.us/j/your-meeting-id",
    "bot_name": "Emotion Analysis Bot"
  }'
```

### 3. Monitor Real-time Data
- **Web Dashboard**: `http://localhost:5000`
- **Participant API**: `http://localhost:5000/api/participants`
- **Individual Stats**: `http://localhost:5000/api/participant/user_001`

## 🌐 API Endpoints

### Participant Management
```http
GET /api/participants
# Returns all participants with current emotional states

GET /api/participant/{id} 
# Returns detailed info for specific participant

GET /api/productivity-stats
# Returns overall meeting productivity metrics
```

### Bot Management
```http
POST /api/send-bot
{
  "meeting_url": "https://zoom.us/j/123456789",
  "bot_name": "Productivity Bot"
}
# Sends Recall.ai bot to join Zoom meeting
```

### Testing
```http
POST /test-numpy-array
# Tests the emotion detection pipeline with sample data
```

## 📊 Real-time Data Flow

### Video Stream Processing
```python
# Continuous stream processing for each participant
for participant_frame in recall_bot_stream:
    participant_id = frame.participant.id
    image_array = convert_base64_to_numpy(frame.data)
    
    # Detect emotions using DeepFace
    emotions = emotion_detector.detect_emotions(
        image_array, 
        participant_metadata
    )
    
    # Update participant tracking
    update_participant_state(participant_id, emotions)
    
    # Broadcast to web clients
    socketio.emit('emotion_update', {
        'participant': participant_id,
        'emotions': emotions,
        'timestamp': datetime.now()
    })
```

### Participant State Tracking
Each participant maintains:
- **Current emotion** (dominant emotion from latest frame)
- **Emotion history** (last 10 emotion detections)
- **Frame count** (total frames processed)
- **Productivity metrics** (calculated from emotion trends)
- **Engagement levels** (based on participation patterns)

## 🧪 Testing

### Run Full Test Suite
```bash
python test_emotion_detection.py
```

### Manual Testing
```bash
# Test single participant
curl http://localhost:5000/test-numpy-array

# Check participant list
curl http://localhost:5000/api/participants

# View productivity stats
curl http://localhost:5000/api/productivity-stats
```

## 🔍 Debugging

### Common Issues

**1. DeepFace Import Error**
```bash
pip install deepface tensorflow
```

**2. OpenCV Issues**
```bash
pip install opencv-python
# or for full features:
pip install opencv-contrib-python
```

**3. WebSocket Connection Problems**
- Check your ngrok URL in `app_ml.py` 
- Verify `WEBSOCKET_PORT` in `.env`
- Ensure firewall allows connections

**4. Recall.ai Bot Issues**
- Verify `RECALL_API_KEY` in `.env`
- Check bot webhook URL matches your ngrok tunnel
- Monitor WebSocket logs for connection status

### Debug Mode
```bash
# Enable detailed logging
python app_ml.py --debug
```

## 📈 Performance Optimization

### For High-Participant Meetings
- **Frame Rate Limiting**: Process every Nth frame to reduce CPU load
- **Batch Processing**: Group multiple participants for efficient GPU usage
- **Emotion Caching**: Cache recent emotions to avoid reprocessing
- **Selective Processing**: Only process frames when faces are detected

### Memory Management
```python
# Automatic cleanup of old participant data
if len(emotion_history[participant_id]) > 10:
    emotion_history[participant_id] = emotion_history[participant_id][-10:]
```

## 🚀 Advanced Features

### Custom Emotion Models
Replace DeepFace with your own models:
```python
# In emotion_detector.py
def detect_emotions(self, image_data, participant_metadata):
    # Use your custom model here
    emotions = your_custom_model.predict(image_data)
    return emotions
```

### LLM Integration for Transcripts
The system includes hooks for LLM-based transcript analysis:
```python
def analyze_transcript_sentiment(self, transcript_text):
    # Integrate with your favorite LLM
    response = llm.analyze_meeting_transcript(transcript_text)
    return response
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Recall.ai](https://recall.ai)** for meeting bot infrastructure
- **[DeepFace](https://github.com/serengil/deepface)** for emotion detection
- **[OpenCV](https://opencv.org/)** for computer vision processing
- **[Flask](https://flask.palletsprojects.com/)** for web framework

---

🎯 **Ready to analyze emotions in your next Zoom meeting!** Start the app and send a bot to see real-time productivity insights.
