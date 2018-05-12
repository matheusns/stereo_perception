import cv2
import numpy as np

def lineDetector(src):
    dst = src.copy()
    edges = cv2.Canny(src,50,150,apertureSize = 3)
    minLineLength = 0
    maxLineGap = 0
    for i in range(300,360):
        lines = cv2.HoughLinesP(edges,1,np.pi/i,100,minLineLength,maxLineGap)
        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(dst,(x1,y1),(x2,y2),(0,0,0),60)
    return dst, edges

def preprocess(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    image = gray.copy()
    image[:] = np.where(gray > 150, 0,gray)
    w = image.shape[1] 
    crop_img = image[200:750, 800:1500]
    lines, edges = lineDetector(crop_img)
    kernel = np.ones((5,5),np.uint8)
    gradient = cv2.morphologyEx(lines, cv2.MORPH_GRADIENT, kernel)
    return gradient, crop_img, lines, edges