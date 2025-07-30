import cv2
import time
import math as m
import mediapipe as mp
from posture_processing import *
import numpy as np
from database import *

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

def give_coords(landmark, frame):
    h, w, _ = frame.shape
    x_coord = int(landmark.x * w)
    y_coord = int(landmark.y * h)
    z_coord = int(landmark.z * w)

    return [x_coord, y_coord, z_coord]

def midpoint(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def distance(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Take live input
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame & detect landmarks
    results_pose = pose_video.process(rgb_frame)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder_coords = give_coords(left_shoulder, frame)
        right_shoulder_coords = give_coords(right_shoulder, frame)
        left_shoulder_text = f"left shoulder x: {left_shoulder_coords[0]} y: {left_shoulder_coords[1]} z: {left_shoulder_coords[2]}"
        right_shoulder_text = f"Right shoulder x: {right_shoulder_coords[0]} y: {right_shoulder_coords[1]} z: {right_shoulder_coords[2]}"
        cv2.putText(frame, left_shoulder_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2)
        cv2.putText(frame, right_shoulder_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2)
        
        left_mouth = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        left_mouth_coords = give_coords(left_mouth, frame)
        right_mouth_coords = give_coords(right_mouth, frame)

        left_mouth_text = f"left mouth x: {left_mouth_coords[0]} y: {left_mouth_coords[1]} z: {left_mouth_coords[2]}"
        right_mouth_text = f"Right mouth x: {right_mouth_coords[0]} y: {right_mouth_coords[1]} z: {right_mouth_coords[2]}"
        
        cv2.putText(frame, left_mouth_text, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2)
        cv2.putText(frame, right_mouth_text, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,225,0), 2)
            

    # Display the processed frame
    cv2.imshow('Pose Detection', frame)

    # Wait for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_video.release()
cv2.destroyAllWindows()