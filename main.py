import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from edge_detection import edge_detect
from region_of_interest import detect_region
from line_display import display_lines
from optimization import avg_lanes



orig_image = cv2.imread('test.jpg')             #import image
my_image = np.copy(orig_image)                  #copy image
edge_image = edge_detect(my_image)              #detect edges
masked_image = detect_region(edge_image)        #region of interest
lane_detect = cv2.bitwise_and(masked_image, edge_image) #detect lane edges

lanes = cv2.HoughLinesP(lane_detect, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
lane_image = display_lines(my_image, lanes)     #detected lanes

detect_image = cv2.addWeighted(my_image, 0.8, lane_image, 1.0, 1)   #original image with detected lanes

optimized_image = avg_lanes(my_image, lanes)    #optimization of lanes
opt_lane_image = display_lines(my_image, optimized_image) #dipsplay of optimized lanes
final_image = cv2.addWeighted(my_image, 0.8, opt_lane_image, 1.0, 1)   

cv2.imshow('Image', final_image)
cv2.waitKey(0)

#plt.imshow(edge_image)
#plt.show()
