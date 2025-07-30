import math
import numpy as np

def calculateAngle(landmark1, landmark2):  

        #landmarks coordinates
        x1, y1 = landmark1
        x2, y2 = landmark2

        # Calculate the angle between the three points
        angle = np.arctan((y2-y1)/(x2-x1))

        # Check if the angle is less than zero
        if angle < 0:
            angle += 360

        return angle
    
def give_coords(landmark, frame):
    h, w, _ = frame.shape
    x_coord = int(landmark.x * w)
    y_coord = int(landmark.y * h)

    return [x_coord, y_coord]

def midpoint(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def distance(p1, p2): # 2 coords (x1, y1) (x2, y2)
    return np.linalg.norm(np.array(p1) - np.array(p2))