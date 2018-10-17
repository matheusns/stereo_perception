# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt 
import math
import plot_cables_extractor as plot_cables

def findContours(src, rgb, new = False):
    (im2, contours, hierarchy) = cv2.findContours(src.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    if new == False:
        return  contours, hierarchy
    else:
        height = src.shape[0]
        width = src.shape[1]
        min_x, min_y = width, height
        max_x = max_y = 0
        dst = src.copy()
        possible_contour = None
        roi = None
        rgb_bounded = None
        img_bounded = None

        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            if h < 130 or w < 90:
                continue

            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)

            possible_contour = contour
            rect = cv2.rectangle(dst, (x,y), (x+w,y+h), (255, 0, 0), 2)
            rgb_bounded = cv2.rectangle(rgb, (x+800,y+200), (x+800+w,y+200+h), (0, 255, 255), 8)
            roi = src[y:y+h, x:x+w]

        return dst,roi, possible_contour, rgb_bounded


def cableExtractor(cnt_full):
    # Percentage of the cable threshold
    cols_percentage = 0.33
    dst_cols = cnt_full.copy()
    
    # Sum of the all coloumns
    sum_cols = np.sum(cnt_full,axis=0)
    sum_cols_max = np.max(sum_cols)
    sum_cols_before = sum_cols

    # Drops all coloumns below ot the threshold
    for i in range (0,len(sum_cols)):
        if sum_cols[i] == sum_cols_max:
            break
        if sum_cols[i] <= sum_cols_max*cols_percentage:
            dst_cols[:,i] = 0

    # This img may contains cable
    dst_before = dst_cols.copy()

    # A new sum is made to remove the cable
    sum_cols = np.sum(dst_cols,axis=0)
    sum_cols_max = np.max(sum_cols)
    last_zero_index = None

    for i in range (0,len(sum_cols)):
        if sum_cols[i] == sum_cols_max:
            # Drops the coloumns before the last zero before the max value
            dst_cols[:, 0:last_zero_index] = 0
            break
        if sum_cols[i] <= sum_cols_max*cols_percentage:
            if sum_cols[i] == 0:
                # Keep save the last zero area
                last_zero_index = i
            dst_cols[:,i] = 0

    last_sum_cols = np.sum(dst_cols,axis=0)

    return dst_cols, sum_cols, sum_cols_before, last_sum_cols, dst_before   

def getFeatures(contour):

    features = []

    (x,y,w,h) = cv2.boundingRect(contour)
    
    aspect_ratio = float(w/float(h))
    features.append(aspect_ratio)

    contour_area = cv2.contourArea(contour)
    features.append(contour_area)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(contour_area)/(hull_area+0.0001)
    features.append(solidity)

    rect_area = w*h
    extent = float(contour_area/float(rect_area))
    features.append(extent)

    perimeter = cv2.arcLength(contour,True)
    features.append(perimeter)

    # Fits an ellipse to the contour         
    # ellipse_copy = cnt_full.copy() 
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    # elipse_img = cv2.ellipse(ellipse_copy,ellipse,255,2)
    a = ma/2
    b = MA/2
    eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
    eccentricity = round(eccentricity/a, 2)

    print "Eccenticity = " + str(eccentricity)

    features.append(eccentricity)

    hull = cv2.convexHull(contour)
    (x_h,y_h,w_h,h_h) = cv2.boundingRect(hull)

    # features = [aspect_ratio, contour_area, solidity, extent, perimeter]

    return features, hull


def img_fill(im_in): 
    # Copy the thresholded image.
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_in | im_floodfill_inv

    return fill_image   


def extractor(mat, crop_img, src, sample, rgb):

    original_mat = mat.copy()
    contours, hierarchy = findContours(src, rgb)  
    contour_img = src.copy()
    elipse_img = src.copy()
    possible_contours = [] 
    roi = None
    hull_area = None
    features_ = []
    key = 0
    eccentricity = None
    

    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)

        if h < 130 or w < 100:
            continue

        img_to_fill = src.copy()
        # Fills the contour area
        cnt_full = img_fill(img_to_fill)

        # image convex hull
        hull = cv2.convexHull(contour)

        # Image without cable
        img_without_cable, cols_sum, cols_sum_before, last_sum_cols, cols_intermedi_img = cableExtractor(cnt_full)

        # contours of the new image
        img_bounded, roi, new_contour, rgb_bounded = findContours(img_without_cable, rgb.copy(), True)

        if roi is not None:

            features_, hull = getFeatures(new_contour)
            features_before, hull_before = getFeatures(contour)

            if features_[5] < 0.94:
                # Extracts the contour features
                print 'features = [aspect_ratio, contour_area, solidity, extent, perimeter, eccentricity]'
                print ''
                print '############### old ##################'
                print features_before
                print '############### new ##################'
                print features_
                print

                dst_2 = img_without_cable.copy()
                hull_img = cv2.drawContours(dst_2, [hull], -1, 255, 5)

                # mat = cv2.cvtColor(mat)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) 
                cnt_full = cv2.cvtColor(cnt_full, cv2.COLOR_GRAY2BGR)
                cols_intermedi_img = cv2.cvtColor(cols_intermedi_img, cv2.COLOR_GRAY2BGR)
                img_without_cable = cv2.cvtColor(img_without_cable, cv2.COLOR_GRAY2BGR)
                hull_img = cv2.cvtColor(hull_img, cv2.COLOR_GRAY2BGR) 
                img_bounded = cv2.cvtColor(img_bounded, cv2.COLOR_GRAY2BGR)
                # rgb_bounded = cv2.cvtColor()

                mat = cv2.resize(mat, (640, 360))
                crop_img = cv2.resize(crop_img, (640, 360))
                src = cv2.resize(src, (640, 360))
                cnt_full = cv2.resize(cnt_full, (640, 360))
                cols_intermedi_img = cv2.resize(cols_intermedi_img, (640, 360))
                img_without_cable = cv2.resize(img_without_cable, (640, 360))
                hull_img = cv2.resize(hull_img, (640, 360))
                img_bounded = cv2.resize(img_bounded, (640, 360))
                rgb_bounded = cv2.resize(rgb_bounded, (640, 360))


                temp = np.vstack([np.hstack([mat, crop_img, src]), np.hstack([cnt_full,cols_intermedi_img, img_without_cable]), np.hstack([hull_img, img_bounded, rgb_bounded]) ])

                temp = cv2.resize(temp, (640,360))

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(temp, str(sample) ,(50,50), font, 1, 255 , 2, cv2.LINE_AA)
                cv2.namedWindow('Depth', cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow('Depth', temp)
                key = cv2.waitKey(1)

                # # Plot the cable cols intensities 
                
                # plot_cables.intensities(cols_sum, cols_sum_before, last_sum_cols)
                
                print sample
            else:
                roi = None
                rgb_bounded = rgb 
                img_bounded = mat 
        else:
            rgb_bounded = rgb
            img_bounded = mat

    return img_bounded, features_, key, rgb_bounded, roi, cnt_full, cols_intermedi_img, img_without_cable
