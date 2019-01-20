import cv2

def edge_detect(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)        #Converting to B&W image
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)         #Blur image to reduce noise
    grad_image = cv2.Canny(blur_image, 60, 140)
    return grad_image