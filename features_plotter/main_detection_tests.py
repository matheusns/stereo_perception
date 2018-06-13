#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:

        # clamp = grampo
        clamp_folder = "/home/matheus/Documents/clamp_knn/" 
        #dumper = amortecedor
        dumper_folder = "/home/matheus/Documents/dumper_knn/"
        # cable = sem obs
        cable_path = "/home/matheus/Documents/cable_knn/" 

        # clamp_folder = "/home/matheus/Documents/plot/clamp/" 
        # # # dumper = amortecedor
        # dumper_folder = "/home/matheus/Documents/plot/dumper/" 
        # cable_path = "/home/matheus/Documents/plot/cable/" 

        # clamp_folder = "/home/matheus/Documents/grampo/depth/"
        # dumper_folder = "/home/matheus/Documents/amor/depth/" 

        file_name = str(sys.argv[0])

    except IndexError:
        print "Error while reading diretoctory path."

    clamp_dirFiles = os.listdir(clamp_folder)
    dumper_dirFiles = os.listdir(dumper_folder)
    cable_dirFiles = os.listdir(cable_path)

    mess = True

    clamp_ordered_files = sorted(clamp_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    dumper_ordered_files = sorted(dumper_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))
    cable_ordered_files = sorted(cable_dirFiles, key=lambda x: (int(re.sub('\D','',x)),x))

    if mess:
        clamp_ordered_files = clamp_dirFiles
        dumper_ordered_files = dumper_dirFiles
        cable_ordered_files = cable_dirFiles

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
    
    cable_sample = []
    cable_aspect_ratio = [] 
    cable_contour_area = [] 
    cable_solidity = []
    cable_extent = []
    cable_perimeter = []
    cable_cont = 0
    cable_eccentricity = []
    all_features = []
    cont_samples = 0


    for j in range (0,len(cable_ordered_files)):

        cont_samples += 1

        print cable_path+cable_ordered_files[j]
        src = cv2.imread(cable_path+cable_ordered_files[j])
        mat = src.copy()

        # if cont_samples == 630:
        #     break

        # cv2.imwrite(cable_path+cont_samples+".png", mat)

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
        cable_eccentricity.append(all_features[5])

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

    print "Media = " + str(sum(cable_eccentricity)/len(cable_eccentricity))

    # plot.vs_feature(dumper_extent, dumper_solidity, clamp_extent, clamp_solidity, cable_extent, cable_solidity, u"Extensão", "Solidez", u"Extensão x Solidez" )
    # plot.vs_feature(dumper_eccentricity, dumper_solidity, clamp_eccentricity, clamp_solidity, cable_eccentricity, cable_solidity, u"Excentricidade", "Solidez", u"Excentricidade x Solidez" )
    # plot.vs_feature(dumper_eccentricity, dumper_extent, clamp_eccentricity, clamp_extent, cable_eccentricity, cable_extent, u"Excentricidade", u"Extensão", u"Excentricidade x Extensão" )
    # plot.vs_feature(dumper_eccentricity, dumper_contour_area, clamp_eccentricity, clamp_contour_area, cable_eccentricity, cable_contour_area, u"Excentricidade", u"Área", u"Excentricidade x Área" )
    # plot.vs_feature(dumper_extent, dumper_contour_area, clamp_extent, clamp_contour_area, cable_extent, cable_contour_area, u"Extensão", u"Área", u"Extensão x Área" )
    # plot.vs_feature(dumper_solidity, dumper_contour_area, clamp_solidity, clamp_contour_area, cable_solidity, cable_contour_area, u"Solidez", u"Área", u"Solidez x Área" )
    plot.threeD(dumper_extent, dumper_solidity, dumper_eccentricity, clamp_extent, clamp_solidity, clamp_eccentricity, cable_extent, cable_solidity, cable_eccentricity, u"Extensão", u"Solidez", u"Excentricidade", u"Extensão x Solidez x Excentricidade")

    # plot.single_feature(dumper_sample, dumper_contour_area, clamp_contour_area, cable_contour_area, u"Área", u"Área")
    # plot.single_feature(dumper_sample, dumper_aspect_ratio, clamp_aspect_ratio, cable_aspect_ratio, u"Razão de Aspecto", u"Feature Razão de Aspecto" )
    # plot.single_feature(dumper_sample, dumper_solidity, clamp_solidity, cable_solidity, "Solidez", u"Medidas de Solidez")
    # plot.single_feature(dumper_sample, dumper_extent, clamp_extent, cable_extent, u"Extensão", u"Medidas de Extensão")
    # plot.single_feature(dumper_sample, dumper_perimeter, clamp_perimeter, cable_perimeter, u"Perímetro", u"Feature Perímetro")
    # plot.single_feature(dumper_sample, dumper_eccentricity, clamp_eccentricity, cable_eccentricity, "Excentricidade", "Excentricidades das amostras analisadas")


    # plt.plot(cable_solidity, ls='-', c = 'teal', alpha = 0.8, linewidth = 2.0, linestyle='-', marker= 's', label="Cable") 

    # plt.plot(dumper_solidity, ls='-', c = 'yellow', alpha = 0.8, linewidth = 2.0, linestyle='-', marker= '^', label="Dumper") 

    # plt.plot(clamp_solidity, ls='-', c = 'orangered', alpha = 0.8, linewidth = 2.0, linestyle='-', marker= '3', label="Clamp") 

    # plt.legend(loc='upper center', scatterpoints = 1, bbox_to_anchor=(0.5, -0.1),  shadow=True, ncol=3)
    # plt.grid(True)
    # plt.show()