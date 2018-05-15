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
        folder = sys.argv[1]
        images_path = "/home/matheus/Documents/"+folder+"/depth/"
        segregation = sys.argv[2]
        anottation = sys.argv[3]
        detector_evaluate =  sys.argv[4]

        if anottation == 'true' or anottation == 't' or anottation == 'T':
            anottation = True
            print "#######################################################"
            print "Annotation Mode"
            print "#######################################################"
        elif anottation == 'false' or anottation == 'f' or anottation == 'F':
            anottation = False

        if segregation == 'true' or segregation == 't' or segregation == 'T':
            segregation = True
            print "#######################################################"
            print "Segregation Mode"
            print "#######################################################"
        elif segregation == 'false' or segregation == 'f' or segregation == 'F':
            segregation = False

        if detector_evaluate == 'true' or detector_evaluate == 't' or detector_evaluate == 'T':
            detector_evaluate = True
            print "#######################################################"
            print "Detector evaluate Mode"
            print "#######################################################"
        elif detector_evaluate == 'false' or detector_evaluate == 'f' or detector_evaluate == 'F':
            detector_evaluate = False
    except IndexError:
        print "Error while reading diretoctory path."
    
    dirFiles = os.listdir(images_path)
    ordered_files = sorted(dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    pos = 0
    neg = 0

    for j in range (0,len(ordered_files) ):

        print images_path+ordered_files[j]
        mat = cv2.imread(images_path+ordered_files[j])

        gradient, crop_img, lines, line_edges = ipp.preprocess(mat)
        img_bounded, contours, roi = features.extractor(gradient, mat)
        # temp = np.vstack([np.hstack([img_bounded, crop_img])])

        # Segregation Mode

        if segregation:
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth', img_bounded)
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == 112:
                cv2.imwrite("/home/matheus/Documents/"+folder+"/with_obst/"+ordered_files[j], mat)
            elif key == 119:
                cv2.imwrite("/home/matheus/Documents/"+folder+"/without_obst/"+ordered_files[j], mat)

        # Annotation Mode

        elif roi is not None and anottation:
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth', roi)
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == 112:
                cv2.imwrite("/home/matheus/Documents/"+folder+"/knn/"+ordered_files[j], mat) # Must be roi_mat

        # Detector Mode

        elif detector_evaluate:
            print "Building..."

    cv2.destroyAllWindows()
    