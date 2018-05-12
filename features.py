import cv2
import numpy as np
import sys

def findContours(src):
    (im2, contours, hierarchy) = cv2.findContours(src.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    return  contours, hierarchy

def extractor(src, rgb):

    contours, hierarchy = findContours(src)  
    height = src.shape[0]
    width = src.shape[1]
    min_x, min_y = width, height
    max_x = max_y = 0
    dst = src.copy()

    possible_contours = [] 


    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        print "object aspect ratio: " + str(float(w/float(h)))
        print "Width = "+str(w)+" Height = " +str(h)
        print "X = "+str(x)+" Y = " +str(y)
        try:
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
        except:
            print "Problem"
        if h < 120 or w < 90:
            continue
        possible_contours.append(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(dst, (x,y), (x+w,y+h), (255, 0, 0), 2)
            
    print "Quantidade de ob = " + str(len(possible_contours))
    # cv2.drawContours(dst2, contours, -1, (255, 0, 0), 2)

    return dst, possible_contours
