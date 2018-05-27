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
import plot_features as plot
import pandas as pd

if __name__ == '__main__':
    try:

        #clamp = grampo
        clamp_folder = "/home/matheus/Documents/clamp_knn/" 
        # clamp_folder = "/home/matheus/Documents/plot/clamp/" 

        file_name = str(sys.argv[1])

    except IndexError:
        print "Error while reading diretoctory path."

    clamp_dirFiles = os.listdir(clamp_folder)
    clamp_ordered_files = sorted(clamp_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    clamp_sample = []
    clamp_aspect_ratio = [] 
    clamp_contour_area = [] 
    clamp_solidity = []
    clamp_extent = []
    clamp_perimeter = []
    clamp_eccentricity = []
    clamp_label = []

    cont_samples = 0

    for j in range (0,len(clamp_ordered_files) ):

        cont_samples += 1

        print clamp_folder+clamp_ordered_files[j]
        src = cv2.imread(clamp_folder+clamp_ordered_files[j])
        mat = src.copy()

        gradient = ipp.preprocess(mat)
        img_bounded, all_features, key = features.extractor(gradient, mat, cont_samples)

        if key == 27:
            cv2.destroyAllWindows()
            break

        clamp_sample.append(cont_samples)
        clamp_aspect_ratio.append(all_features[0])  
        clamp_contour_area.append(all_features[1])
        clamp_solidity.append(all_features[2])
        clamp_extent.append(all_features[3])
        clamp_perimeter.append(all_features[4])
        clamp_eccentricity.append(all_features[5])

    print ''
    print '=================================================='
    print "Clamp samples len = " + str( len(clamp_sample) )
    print '=================================================='
    print ''

    print ''
    print '=================================================='
    print "Clamp Vector  len = " + str( len(clamp_contour_area) )
    print '=================================================='
    print ''

    clamp_label = np.zeros(len(clamp_sample), dtype=int)
    areas = clamp_contour_area 
    aspects = clamp_aspect_ratio 
    solidities = clamp_solidity
    extents = clamp_extent 
    perimeters = clamp_perimeter
    eccentricities = clamp_eccentricity
    labels =  clamp_label

    print "Samples = " + str(clamp_sample)


    data_frame = { 'area': areas, 'aspects': aspects, 'solidities': solidities, 'extents': extents, 'perimeters': perimeters, 'eccentricities': eccentricities, 'labels': labels}

    df = pd.DataFrame(data_frame, columns = ['area', 'aspects', 'solidities', 'extents', 'perimeters', 'eccentricities', 'labels'])

    # df.to_csv(file_name + '.csv')
    df.to_csv(file_name + '.txt')