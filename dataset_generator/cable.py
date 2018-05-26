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

        # cable = sem obs
        cable_path = "/home/matheus/Documents/cable_knn/" 
        # cable_path = "/home/matheus/Documents/plot/cable/" 

        file_name = str(sys.argv[1])

    except IndexError:
        print "Error while reading diretoctory path."

    cable_dirFiles = os.listdir(cable_path)
    cable_ordered_files = sorted(cable_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    
    cable_sample = []
    cable_aspect_ratio = [] 
    cable_contour_area = [] 
    cable_solidity = []
    cable_extent = []
    cable_perimeter = []
    cable_cont = 0
    all_features = []
    cont_samples = 0


    for j in range (0,len(cable_ordered_files)):

        cont_samples += 1

        print cable_path+cable_ordered_files[j]
        src = cv2.imread(cable_path+cable_ordered_files[j])
        mat = src.copy()

        print mat.shape

        gradient = ipp.preprocess(mat, False)
        img_bounded, all_features, key = features.extractor(gradient, mat, cont_samples, True)
        if key == 27:
            cv2.destroyAllWindows()
            break

        cable_sample.append(cont_samples)
        cable_aspect_ratio.append(all_features[0])  
        cable_contour_area.append(all_features[1])
        cable_solidity.append(all_features[2])
        cable_extent.append(all_features[3])
        cable_perimeter.append(all_features[4])

    print ''
    print '=================================================='
    print "Cable samples len = " + str( len(cable_sample) )
    print '=================================================='
    print ''

    print ''
    print '=================================================='
    print "Cable Vector len = " + str( len(cable_sample) )
    print '=================================================='
    print ''

    cable_label = np.ones(len(cable_sample), dtype=int)*2
    
    areas = cable_contour_area
    aspects = cable_aspect_ratio
    solidities = cable_solidity
    extents = cable_extent
    perimeters = cable_perimeter
    labels =  cable_label


    # Pandas data frame
    data_frame = { 'area': areas, 'aspects': aspects, 'solidities': solidities, 'extents': extents, 'perimeters': perimeters, 'labels': labels}

    df = pd.DataFrame(data_frame, columns = ['area', 'aspects', 'solidities', 'extents', 'perimeters', 'labels'])

    # df.to_csv(file_name + '.csv')
    df.to_csv(file_name + '.txt')