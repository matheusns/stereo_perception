"""
   main.py
   Script to read images and process them.   
   Matheus Nascimento
   April, 2018
   Based on Python 2.7 Version    
"""
import os
import cv2
import numpy as np
import sys
import re

if __name__ == '__main__':
    try:
        images_path = sys.argv[1]
    except IndexError:
        images_path = "/media/matheus/MULTIBOOT/data_acquisition/05_05_11110/depth/"
    
    dirFiles = os.listdir(images_path)
    ordered_files = sorted(dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    pos = 0
    neg = 0

    for j in range (0,len(ordered_files) ):

        print images_path+ordered_files[j]
        mat = cv2.imread(images_path+ordered_files[j], -1)

        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

        image = gray.copy()

        image[:] = np.where(gray > 150, 0,gray)

        crop_img = image[1:506,:]


        kernel = np.ones((5,5),np.uint8)
        gradient = cv2.morphologyEx(crop_img, cv2.MORPH_GRADIENT, kernel)
        # canny_img = cv2.Canny(gradient, 20, 230)
        # opening = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)

        # dilation = cv2.dilate(mat,kernel,iterations = 1)

        (im2, contours, hierarchy) = cv2.findContours(gradient.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []

        height, width, channels = mat.shape

        min_x, min_y = width, height
        max_x = max_y = 0

        dst = crop_img.copy()
        # dst2 = crop_img.copy()

        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            print "object aspect ratio: " + str(w/h)

            print "Width = "+str(w)+" Height = " +str(h)
            print "X = "+str(x)+" Y = " +str(y)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
            if w > 80 and h > 80:
                cv2.rectangle(dst, (x,y), (x+w,y+h), (255, 0, 0), 2)

        if max_x - min_x > 0 and max_y - min_y > 0:
            cv2.rectangle(dst, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

        print "Quantidade de ob = " + str(len(contours))

        # cv2.drawContours(dst2, contours, -1, (255, 0, 0), 2)

        if (len(contours) == 1):
            pos +=1
        else:
            neg +=1

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break        

        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Quantidade de objetos: "+str(len(objetos)), cv2.WINDOW_NORMAL)

        temp = np.vstack([np.hstack([dst, crop_img])])
        # cv2.imshow('Depth', image)
        cv2.imshow('Depth', dst)

        print "Pos = " + str(pos)+" Neg = " +str(neg)

    cv2.destroyAllWindows()
    