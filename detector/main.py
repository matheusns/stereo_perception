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
import detector_features
import knn_predictor as knn

# python main.py 05_05_2320 t f f clamp

if __name__ == '__main__':
    try:
        folder = sys.argv[1]
        images_path = "/home/matheus/Documents/"+folder+"/depth/"
        normalized_path = "/home/matheus/Documents/"+folder+"/normalized/"
        segregation = sys.argv[2]
        anottation = sys.argv[3]
        detector_evaluate =  sys.argv[4]
        obstacle = sys.argv[5]

        if anottation == 'true' or anottation == 't' or anottation == 'T':
            anottation = True

        elif anottation == 'false' or anottation == 'f' or anottation == 'F':
            anottation = False

        if segregation == 'true' or segregation == 't' or segregation == 'T':
            segregation = True

        elif segregation == 'false' or segregation == 'f' or segregation == 'F':
            segregation = False

        if detector_evaluate == 'true' or detector_evaluate == 't' or detector_evaluate == 'T':
            detector_evaluate = True

        elif detector_evaluate == 'false' or detector_evaluate == 'f' or detector_evaluate == 'F':
            detector_evaluate = False

    except IndexError:
        print "Error while reading diretoctory path."

    dirFiles = os.listdir(images_path)
    dirFiles_normalized = os.listdir(normalized_path)

    ordered_files = sorted(dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    area_vector_value = []
    sample = []
    cont_samples = 0
    detect_with = 0
    detect_without = 0
    not_detect_with = 0
    not_detect_without = 0
    detector_samples = 0

    for j in range (0,len(ordered_files) ):

        cont_samples += 1
        print images_path+ordered_files[j]
        src = cv2.imread(images_path+ordered_files[j])
        normalized = cv2.imread(normalized_path+ordered_files[j])
        mat = src.copy()

        gradient, without_close, test = ipp.preprocess(mat)
        img_bounded, all_features, key, rgb_bounded, roi, flood, cols_intermedi_img, without_cable = detector_features.extractor(mat, without_close, gradient, cont_samples, normalized)

        # Segregation Mode
        if segregation:

            print "Key = " + str(key)
            if key == 27:
                cv2.destroyAllWindows()
                break

            elif key == 83 or key == 115:
                detector_samples += 1
                # Clamp
                if obstacle == "clamp":
                    cv2.imwrite("/home/matheus/Documents/tcc/"+"final_cable_"+ordered_files[j], without_cable)
                    cv2.imwrite("/home/matheus/Documents/tcc/"+"flood_"+ordered_files[j], flood)
                # Dumper
                elif obstacle == "dumper":
                    cv2.imwrite("/home/matheus/Documents/dumper_knn/"+ordered_files[j], mat)
                elif obstacle == "cable":
                    print "==== == Cable Saved at /home/matheus/Documents/cable_knn/"+ordered_files[j] + "== ===="
                    cv2.imwrite("/home/matheus/Documents/cable_knn/"+ordered_files[j], gradient)
                    print ''

            if detector_samples == 110:
                break

        # Annotation Mode
        elif roi is not None and anottation:
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth', roi)
            key = cv2.waitKey(0)

            # If 'esc' is pressed.
            if key == 27:
                cv2.destroyAllWindows()
                break
            # If 'p' is pressed.
            elif key == 112:
                cont_samples+=1
                sample.append(cont_samples)
                area_vector_value.append(contour_area)
                print "cont len = " + str(len(sample)) + "cont area = " + str(len(area_vector_value))
                if cont_samples == 510:
                    break
                cv2.imwrite("/home/matheus/Documents/"+folder+"/knn_train/"+ordered_files[j], roi) # Must be roi_mat
                cv2.imwrite("/home/matheus/Documents/"+folder+"/with_obst/"+ordered_files[j], mat) # Must be roi_mat
            # If 'w' is pressed.
            elif key == 119:
                cv2.imwrite("/home/matheus/Documents/"+folder+"/knn_train_without/"+ordered_files[j], roi) # Must be roi_mat
                cv2.imwrite("/home/matheus/Documents/"+folder+"/without_obst/"+ordered_files[j], mat) # Must be roi_mat

        #Detector evaluator
        elif detector_evaluate:
            cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
            # temp = np.vstack([np.hstack([img_bounded, crop_img])])
            resized_image = cv2.resize(img_bounded, (640, 360))
            cv2.imshow('Depth', resized_image)

            key = cv2.waitKey(0)
            print "Key = " +str(key)
            if key == 27:
                cv2.destroyAllWindows()
                break
            elif key == 67 or key == 99:
                detect_with += 1
                detector_samples += 1
                cv2.imwrite("/home/matheus/Documents/knn/"+folder+"/detected/"+ordered_files[j], mat)
            elif key == 83 or key == 115:
                detector_samples += 1
                not_detect_without += 1
                cv2.imwrite("/home/matheus/Documents/knn/"+folder+"/not_detected/"+ordered_files[j], mat)
            elif key == 74 or key == 106:
                detector_samples += 1
                detect_without += 1
            elif key == 66 or key == 98:
                detector_samples += 1
                not_detect_with += 1

            if detector_samples == 500:
                break

            print "Detected with = " + str(detect_with)
            print "Detected Without = " + str(detect_without)
            print "Not Detected with = " + str(not_detect_with)
            print "Not Detected Without = " + str(not_detect_without)

    if detector_evaluate:
        print "Detected with = " + str(detect_with)
        print "Detected Without = " + str(detect_without)
        print "Not Detected with = " + str(not_detect_with)
        print "Not Detected Without = " + str(not_detect_without)
    cv2.destroyAllWindows()
