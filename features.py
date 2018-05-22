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
    roi = None
    rgb_bounded = rgb.copy()
    hull_area = None

    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        # Extrair as metricas e buscar pela razao de aspecto
        # print "object aspect ratio: " + str(float(w/float(h)))
        # print "Width = "+str(w)+" Height = " +str(h)
        # print "X = "+str(x)+" Y = " +str(y)
        try:
            # aspect = cv2.
            contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(contour_area)/hull_area
        except:
            print "Problem"
            roi = None
            return

        if h < 120 or w < 90:
            continue
        possible_contours.append(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            print "P_rect = ("+str(y)+","+str(x)+")"
            print "hull area = "+str(hull_area)
            print "contour area = "+str(contour_area)
            rect = cv2.rectangle(dst, (x,y), (x+w,y+h), (255, 0, 0), 2)
            rgb_bounded = cv2.rectangle(rgb, (x+800,y+200), (x+800+w,y+200+h), (255, 0, 0), 5)
            roi = src[y:y+h, x:x+w]
            
            
    # print "Quantidade de ob = " + str(len(possible_contours))
    # cv2.drawContours(dst2, contours, -1, (255, 0, 0), 2)

    return dst, possible_contours, roi, rgb_bounded, hull_area
