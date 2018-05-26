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
        #dumper = amortecedor
        dumper_folder = "/home/matheus/Documents/dumper_knn/"
        # cable = sem obs
        cable_path = "/home/matheus/Documents/cable_knn/" 

        # clamp_folder = "/home/matheus/Documents/plot/clamp/" 
        # # # dumper = amortecedor
        # dumper_folder = "/home/matheus/Documents/plot/dumper" 
        # cable_path = "/home/matheus/Documents/plot/cable/" 

        clamp_path = "/home/matheus/Documents/grampo/depth/"
        dumper_path = "/home/matheus/Documents/amor/depth/" 

        file_name = str(sys.argv[0])

    except IndexError:
        print "Error while reading diretoctory path."

    clamp_dirFiles = os.listdir(clamp_folder)
    dumper_dirFiles = os.listdir(dumper_folder)
    cable_dirFiles = os.listdir(cable_path)

    clamp_ordered_files = sorted(clamp_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    dumper_ordered_files = sorted(dumper_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    cable_ordered_files = sorted(cable_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    

    clamp_sample = []
    clamp_aspect_ratio = [] 
    clamp_contour_area = [] 
    clamp_solidity = []
    clamp_extent = []
    clamp_perimeter = []
    clamp_label = []

    cont_samples = 0

    for j in range (0,len(clamp_ordered_files) ):

        cont_samples += 1

        print clamp_path+clamp_ordered_files[j]
        src = cv2.imread(clamp_path+clamp_ordered_files[j])
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

        

    dumper_sample = []
    dumper_aspect_ratio = [] 
    dumper_contour_area = [] 
    dumper_solidity = []
    dumper_extent = []
    dumper_perimeter = []
    dumper_cont = 0
    all_features = []
    cont_samples = 0

    for j in range (0,len(dumper_ordered_files) ):

        cont_samples += 1

        print dumper_path+dumper_ordered_files[j]
        src = cv2.imread(dumper_path+dumper_ordered_files[j])
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
    print "Dumper samples len = " + str( len(dumper_sample) )
    print "Clamp samples len = " + str( len(clamp_sample) )
    print "Cable samples len = " + str( len(cable_sample) )
    print '=================================================='
    print ''

    print ''
    print '=================================================='
    print "Dumper Vector len = " + str( len(dumper_contour_area) )
    print "Clamp Vector  len = " + str( len(clamp_contour_area) )
    print "Cable Vector len = " + str( len(cable_sample) )
    print '=================================================='
    print ''

    plot.area(dumper_sample, dumper_contour_area, clamp_contour_area, cable_contour_area)

    plot.aspect_ratio(dumper_sample, dumper_aspect_ratio, clamp_aspect_ratio, cable_aspect_ratio)

    plot.solidity(dumper_sample, dumper_solidity, clamp_solidity, cable_solidity)

    plot.extent(dumper_sample, dumper_extent, clamp_extent, cable_extent)

    plot.perimeter(dumper_sample, dumper_perimeter, clamp_perimeter, cable_perimeter)