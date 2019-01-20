import cv2
import numpy as np

def detect_region(image):
    height = image.shape[0]                                     #shape of the image
    detect_triangle = np.array([[(200,height), (1100, height), (550,250)]])    #tiangle zone of detection
    mask = np.zeros_like(image)           #masking with zero arrays same as original image
    cv2.fillPoly(mask, detect_triangle, 255)
    return mask