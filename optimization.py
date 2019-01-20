import cv2
import numpy as np 

def make_coordinates(image, line_param):
    slope, intercept = line_param
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def avg_lanes(image, lines):
    left_lane = []
    right_lane = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_lane.append((slope, intercept))
        else:
            right_lane.append((slope, intercept))

    #print(left_lane)
    #print(right_lane)
    left_lane_avg = np.average(left_lane, axis=0)
    right_lane_avg = np.average(right_lane, axis=0)
    left_lane_out = make_coordinates(image, left_lane_avg)
    right_lane_out = make_coordinates(image, right_lane_avg)

    return np.array([left_lane_out, right_lane_out])