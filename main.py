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
import imgpreprocess as ipp
import features
import floodfill as ff

if __name__ == '__main__':
    try:
        images_path = sys.argv[1]
    except IndexError:
        images_path = "/home/matheus/Documents/amortecedor/depth/"
    
    dirFiles = os.listdir(images_path)
    ordered_files = sorted(dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    pos = 0
    neg = 0

    for j in range (0,len(ordered_files) ):

        print images_path+ordered_files[j]
        mat = cv2.imread(images_path+ordered_files[j])
        gradient, crop_img, lines, line_edges = ipp.preprocess(mat)
        img_bounded, contours = features.extractor(gradient, mat)
        if (len(contours) > 1):
            neg +=1
        else:
            pos +=1
        # temp = np.vstack([np.hstack([img_bounded, crop_img])])
        if lines is not None:
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth', img_bounded)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break        
        print "Pos = " + str(pos)+" Neg = " +str(neg)

    cv2.destroyAllWindows()
    