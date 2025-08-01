import cv2
import time
import math as m
import mediapipe as mp
from posture_processing import *
import numpy as np
from posture_database import *
from utils import *

#db = database()

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
def posture(frame, db, participant_name, trigger_llm_callback=None):
    # ok, frame = camera_video.read()
    # if not ok:
    #     continue
    
    # frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame & detect landmarks
    results_pose = pose_video.process(rgb_frame)

    if results_pose.pose_landmarks:
        # mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        db.calculateValues(results_pose, results_pose, mp_pose, frame)
        
        left_shoulder = db.get_left_shoulder()
        right_shoulder = db.get_right_shoulder()
        mouth = db.get_mouth()
        shoulder_angle = db.get_shoulder_angle()
        status_tilt = db.get_status_tilt()
        status_sm_dist = db.get_status_sm_dist()
        last_status_tilt = db.get_last_status_tilt()
        last_status_sm_dist = db.get_last_status_sm_dist()
        main_status = db.get_main_status()

        # Ultimate Posture Warning
        if db.get_main_status() != "Great Posture! Keep it up!":
            if  not db.get_ultimate_warning() and time.time() - db.get_time_bad_posture() > 60:
                print("EMERGENCY ALERT. TERRIBLE TERRIBLE POSTURE FOR TOO LONG. LIFE THREATENING. FIX OR ELSE")
                db.set_ultimate_warning(True)
        else:
            db.set_ultimate_warning(False)
            db.update_time_bad_posture()
            

        if db.get_status_tilt() != db.get_last_status_tilt() or db.get_status_sm_dist() != db.get_last_status_sm_dist():
            if not db.get_triggered():
                db.set_triggered(True)
                db.update_time()  # Start timing the change
            elif time.time() - db.get_time() > 10:
                db.update_last_status()
                db.set_triggered(False)
                if trigger_llm_callback:
                    trigger_llm_callback("", db.get_main_status(), participant_name)
        else:
            db.set_triggered(False)
        

        # if left_shoulder:
        #     text = f"Left Shoulder: x: {int(left_shoulder[0])}, y: {int(left_shoulder[1])}"
        #     cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # if right_shoulder:
        #     text = f"Right Shoulder: x: {int(right_shoulder[0])}, y: {int(right_shoulder[1])}"
        #     cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # if mouth:
        #     text = f"Mouth Center: x: {int(mouth[0])}, y: {int(mouth[1])}"
        #     cv2.putText(frame, text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # if shoulder_angle:
        #     text = f"Shoulder Angle: {int(shoulder_angle)}"
        #     cv2.putText(frame, text, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # text = f"Tilt - Last: {last_status_tilt} Current: {status_tilt}"
        # cv2.putText(frame, text, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # text = f"Dist - Last: {last_status_sm_dist} Current: {status_sm_dist}"
        # cv2.putText(frame, text, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # if main_status:
        #     text = f"Main Status: {main_status}"
        #     cv2.putText(frame, text, (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        left_mouth = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
    # Display the processed frame
    #cv2.imshow('Pose Detection', frame)

    # Wait for 'q' key to exit the loop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    print(main_status)

# camera_video.release()
# cv2.destroyAllWindows()