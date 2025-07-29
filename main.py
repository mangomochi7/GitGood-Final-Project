import cv2
import time
import math as m
import mediapipe as mp
from posture_processing import *
import numpy as np

mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils 

# Initialize camera input
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 880)  # Set camera frame width to 880px
camera_video.set(4, 660)  # Set camera frame height to 660px

# Take live input
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame & detect landmarks
    results = pose_video.process(rgb_frame)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the processed frame
    cv2.imshow('Pose Detection', frame)

    # Wait for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def midpoint(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return ((p1[0] + p1[0]) / 2, (p2[1] + p2[1]) / 2)

def distance(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return np.linalg.norm(np.array(p1) - np.array(p2))

camera_video.release()
cv2.destroyAllWindows()