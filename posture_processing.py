import numpy as np
import math
from database import *
from utils import *
import mediapipe as mp
from database import database

def check_posture(user_database: database):
    issues_list  = []
    nice_list = []

    left_shoulder = user_database.left_shoulder
    right_shoulder = user_database.right_shoulder
    mouth = user_database.mouth

    shoulder_angle = user_database.shoulder_angle

    if shoulder_angle  > 10:
        issues_list.append("tilted shoulders")
    else:
        nice_list.append("shoulders are level")
    


    

    

    
    
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



