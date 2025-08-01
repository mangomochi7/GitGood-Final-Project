import mediapipe as mp
import math
import numpy as np
import time

import posture_processing
from utils import *
from posture_processing import *

class database:

    def __init__(self):
        self.right_shoulder = None
        self.left_shoulder = None
        self.shoulder_angle = None
        self.mouth = None
        self.time = time.time()

        self.time_bad_posture = time.time()

        self.status_tilt = False
        self.status_sm_dist = False
        self.last_status_tilt = False
        self.last_status_sm_dist = False

        self.triggered = False
        self.ultimate_warning = False

        self.main_status = "Great Posture! Keep it up!"
        self.color = (0, 0, 255)  # BGR

        self.counter = 0

    def calculateValues(self, landmarks, pose_landmarks, mp_pose, frame):
        self.left_shoulder = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]), frame)
        self.right_shoulder = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]), frame)
        self.shoulder_angle = calculateAngle(self.right_shoulder, self.left_shoulder)
        
        left_mouth = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]), frame)
        right_mouth = give_coords((pose_landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]), frame)
        self.mouth = midpoint(left_mouth, right_mouth)

        if self.last_status_tilt and self.last_status_sm_dist:
            self.main_status = "MAJOR Posture ALERT! Keep your shoulders straight and sit up!"
        elif self.last_status_tilt:
            self.main_status = "Make sure to keep your shoulders straight!"
        elif self.last_status_sm_dist:
            self.main_status = "Make sure to sit up straight!"
        else:
            self.main_status = "Great Posture! Keep it up!"
            
        posture_processing.check_posture(self)
        
    def get_triggered(self):
        return self.triggered

    def set_triggered(self, boolean):
        self.triggered = boolean

    def update_last_status(self):
        self.last_status_tilt = self.status_tilt
        self.last_status_sm_dist = self.status_sm_dist

    def get_left_shoulder(self):
        return self.left_shoulder

    def get_right_shoulder(self):
        return self.right_shoulder

    def get_shoulder_angle(self):
        return self.shoulder_angle

    def get_mouth(self):
        return self.mouth
    
    def get_status_tilt(self):
        return self.status_tilt
    
    def get_status_sm_dist(self):
        return self.status_sm_dist
    
    def set_status_tilt(self, boolean):
        self.status_tilt = boolean

    def set_status_sm_dist(self, boolean):
        self.status_sm_dist = boolean

    def update_time(self):
        self.time = time.time()
    
    def get_time(self):
        return self.time
    
    def update_time_bad_posture(self):
        self.time_bad_posture = time.time()

    def get_time_bad_posture(self):
        return self.time_bad_posture
    
    def get_last_status_sm_dist(self):
        return self.last_status_sm_dist
    
    def get_last_status_tilt(self):
        return self.last_status_tilt
    
    def get_main_status(self):
        return self.main_status
    
    def set_ultimate_warning(self, boolean):
        self.ultimate_warning = boolean

    def get_ultimate_warning(self):
        return self.ultimate_warning
    
    def add_counter(self):
        self.counter +=1
    
    def reset_counter(self):
        self.counter = 0