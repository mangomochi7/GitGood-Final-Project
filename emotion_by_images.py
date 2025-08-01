import cv2
from deepface import DeepFace

def get_emotion(frame, counter):
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(60, 60), maxSize=(500, 500))
    
    emotions = []
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]
        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        emotions.append(emotion)

    primary_emotion = emotions[0] if emotions else 'neutral'

    # if the previous frame triggered the llm, reset
    if counter.is_triggered():
        print(f"🔄 Resetting trigger (was triggered last frame)")
        counter.reset_trigger()

    # if a negative emotion is detected
    if primary_emotion != 'neutral' and primary_emotion != 'happy':
        counter.negative_emotion() # increase number of consecutive negative frames, set self.negative to True
        frames = counter.get_frames() # gets number of consecutive negative frames
        print(f"📊 Negative emotion count: {frames}, Cooldown: {counter.get_trigger_cooldown()}")
        if frames >= 5 and counter.get_trigger_cooldown() > 10: # if there have been more than 5 consecutive frames and it has been 10 frames since the last trigger
            print(f"🚨 TRIGGERING! {frames} consecutive negative emotions")
            counter.trigger()
        else:
            counter.not_triggered() # increase frames since last trigger
    else: # a non-negative emotion is detected
        counter.negative_end() # self.negative is False, consecutive negative frames set to 0
        counter.not_triggered()

    return primary_emotion


class EmotionCounter:

    def __init__(self):
        self.negative_frames = 0
        self.negative = False
        self.triggered = False
        self.frames_since_trigger = 0

    def negative_emotion(self):
        self.negative_frames += 1
        self.negative = True

    def get_frames(self):
        return self.negative_frames

    def negative_end(self):
        self.negative = False
        self.negative_frames = 0

    def trigger(self):
        self.triggered = True
        self.frames_since_trigger = 0

    def is_triggered(self):
        return self.triggered

    def isTriggered(self):
        return self.triggered

    def reset_trigger(self):
        self.triggered = False

    def not_triggered(self):
        self.frames_since_trigger += 1

    def get_trigger_cooldown(self):
        return self.frames_since_trigger

    