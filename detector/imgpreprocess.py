import cv2
import numpy as np

def preprocess(src, crop = True):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    test = gray.copy()
    image = gray.copy()
    image[:] = np.where(gray > 0, 255,0)
    w = image.shape[1]
    kernel = np.ones((15,15),np.uint8)
    if crop:
        image = image[200:750, 800:1500]
        test = test[200:750, 800:1500]
        without_close = image
    gradient = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return gradient, without_close, test