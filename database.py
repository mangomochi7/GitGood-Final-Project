import mediapipe as mp
import math
import numpy as np
from utils import *

class database:

    def __init__(self):
        self.right_shoulder = None
        self.left_shoulder = None
        self.shoulder_angle = None
        self.mouth = None

        self.status = 'Unknown Posture'
        self.color = (0, 0, 255)  # BGR

    def calculateValues(self, landmarks, pose_landmarks, mp_pose, frame):
        self.left_shoulder = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]), frame)
        self.right_shoulder = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]), frame)
        self.shoulder_angle = calculateAngle(self.left_shoulder, self.right_shoulder)
        
        left_mouth = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]), frame)
        right_mouth = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]), frame)
        self.mouth = midpoint(left_mouth, right_mouth)

    
        
    def get_left_shoulder(self):
        return self.left_shoulder

    def get_right_shoulder(self):
        return self.right_shoulder

    def get_mouth(self):
        return self.mouth