import numpy as np
import math
from database import *
from utils import *
import mediapipe as mp
from database import database

SHOULDER_ANGLE_THRESHOLD = 7
THRESHOLD_RATIO = 0.4

def check_posture(user_database: database):
    issues_list = []
    nice_list = []

    left_shoulder = user_database.get_left_shoulder()
    right_shoulder = user_database.get_right_shoulder()
    mouth = user_database.get_mouth()

    shoulder_angle = user_database.shoulder_angle
    #shoulder tilt stuff
    if shoulder_angle > SHOULDER_ANGLE_THRESHOLD:
        user_database.set_status_tilt(True)
    else:
        user_database.set_status_tilt(False)

    #mouth to shoulder distance
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    chin_distance = distance(mouth, shoulder_mid)
    shoulder_distance = distance(left_shoulder, right_shoulder)

    dist_ratio = chin_distance/shoulder_distance

    if dist_ratio < THRESHOLD_RATIO:
        user_database.set_status_sm_dist(True)
    else:
        user_database.set_status_sm_dist(False)

    

        


    

    

    
    
# def checkIfIdle(self):
#         # 1. if legs are straight - best = 180 degrees
#         if ((200 > self.right_knee_angle > 160) and
#                 (200 > self.left_knee_angle > 160) and
#                 # 2. if hands are down/to legs - best = 100 degrees (shoulders)
#                 (120 > self.right_shoulder_angle > 80) and
#                 (120 > self.left_shoulder_angle > 80) and
#                 # 3. if legs are together - best = 90 degrees (hips)
#                 (110 > self.right_hip_angle > 70) and
#                 (110 > self.left_hip_angle > 70)):
#             self.status = 'Idle'
#             self.color = (123, 221, 97)
#         return self.status, self.color



