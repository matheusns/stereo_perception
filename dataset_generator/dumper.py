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

        #dumper = amortecedor
        dumper_folder = "/home/matheus/Documents/dumper_knn/"
        # dumper_folder = "/home/matheus/Documents/plot/dumper" 

        file_name = str(sys.argv[1])

    except IndexError:
        print "Error while reading diretoctory path."

    dumper_dirFiles = os.listdir(dumper_folder)
    dumper_ordered_files = sorted(dumper_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    
    dumper_sample = []
    dumper_aspect_ratio = [] 
    dumper_contour_area = [] 
    dumper_solidity = []
    dumper_extent = []
    dumper_perimeter = []
    dumper_eccentricity = []
    dumper_cont = 0
    all_features = []
    cont_samples = 0

    for j in range (0,len(dumper_ordered_files) ):

        cont_samples += 1

        print dumper_folder+dumper_ordered_files[j]
        src = cv2.imread(dumper_folder+dumper_ordered_files[j])
        mat = src.copy()

        gradient = ipp.preprocess(mat)
        img_bounded, all_features, key = features.extractor(gradient, mat, cont_samples)

        if key == 27:
            cv2.destroyAllWindows()
            break
        
        dumper_sample.append(cont_samples)
        dumper_aspect_ratio.append(all_features[0])  
        dumper_contour_area.append(all_features[1])
        dumper_solidity.append(all_features[2])
        dumper_extent.append(all_features[3])
        dumper_perimeter.append(all_features[4])
        dumper_eccentricity.append(all_features[5])

        

    print ''
    print '=================================================='
    print "Dumper samples len = " + str( len(dumper_sample) )
    print '=================================================='
    print ''

    print ''
    print '=================================================='
    print "Dumper Vector len = " + str( len(dumper_contour_area) )
    print '=================================================='
    print ''

    dumper_label = np.ones(len(dumper_sample), dtype=int)
    
    areas = dumper_contour_area
    aspects = dumper_aspect_ratio
    solidities = dumper_solidity
    extents = dumper_extent
    perimeters = dumper_perimeter
    eccentricities = dumper_eccentricity
    labels =  dumper_label

    data_frame = { 'area': areas, 'aspects': aspects, 'solidities': solidities, 'extents': extents, 'perimeters': perimeters, 'eccentricities': eccentricities, 'labels': labels}

    df = pd.DataFrame(data_frame, columns = ['area', 'aspects', 'solidities', 'extents', 'perimeters', 'eccentricities', 'labels'])

    # df.to_csv(file_name + '.csv')
    df.to_csv(file_name + '.txt')