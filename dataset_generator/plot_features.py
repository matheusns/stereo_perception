#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib as mat
import matplotlib.pyplot as plt

fontsize = 18
mat.rc('legend', fontsize=fontsize, handlelength=3)
mat.rc('axes', titlesize=fontsize)
mat.rc('axes', labelsize=25)
mat.rc('xtick', labelsize=fontsize)
mat.rc('ytick', labelsize=fontsize)
# mat.rc('text', usetex=True)
mat.rc('font', size=fontsize, family='serif', style='normal', variant='normal',stretch='normal', weight='normal')

def area(x,y1,y2,y3):

    plt.plot(x, y1, ls='-', c = 'blue', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y1,color='blue')
    
    plt.plot(x, y2, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y2,color='red')

    plt.plot(x, y3, ls='-', c = 'green', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y3,color='green')

    x_axis = [0, 510]

    plt.xlim(0, 510)
    # plt.xticks( np.arange(min(y), max(y), 25) )
    # plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
    plt.xlabel(u"Amostras")
    plt.ylabel(u"Área")

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(custom_lines, ['Dumper', 'Clamp', 'Cable'], loc='b')
    # plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
    # Grid
    plt.grid(True)
    plt.show()

def aspect_ratio(x,y1,y2,y3):

    plt.plot(x, y1, ls='-', c = 'blue', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y1,color='blue')
    
    plt.plot(x, y2, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y2,color='red')

    plt.plot(x, y3, ls='-', c = 'green', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y3,color='green')

    x_axis = [0, 510]

    plt.xlim(0, 510)
    # plt.xticks( np.arange(min(y), max(y), 25) )
    # plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
    plt.xlabel(u"Amostras")
    plt.ylabel(u"Razão de Aspecto")

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(custom_lines, ['Dumper', 'Clamp', 'Cable'], loc='b')
    # plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
    # Grid
    plt.grid(True)
    plt.show()

def solidity(x,y1,y2,y3):

    plt.plot(x, y1, ls='-', c = 'blue', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y1,color='blue')
    
    plt.plot(x, y2, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y2,color='red')

    plt.plot(x, y3, ls='-', c = 'green', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y3,color='green')

    x_axis = [0, 510]

    plt.xlim(0, 510)
    # plt.xticks( np.arange(min(y), max(y), 25) )
    # plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
    plt.xlabel(u"Amostras")
    plt.ylabel(u"Solidez")

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(custom_lines, ['Dumper', 'Clamp', 'Cable'], loc='b')
    # plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
    # Grid
    plt.grid(True)
    plt.show()

def extent(x,y1,y2,y3):

    plt.plot(x, y1, ls='-', c = 'blue', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y1,color='blue')
    
    plt.plot(x, y2, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y2,color='red')

    plt.plot(x, y3, ls='-', c = 'green', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y3,color='green')

    x_axis = [0, 510]

    plt.xlim(0, 510)
    # plt.xticks( np.arange(min(y), max(y), 25) )
    # plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
    plt.xlabel(u"Amostras")
    plt.ylabel(u"Extensão")

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(custom_lines, ['Dumper', 'Clamp', 'Cable'])
    # plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
    # Grid
    plt.grid(True)
    plt.show()

def perimeter(x,y1,y2,y3):

    plt.plot(x, y1, ls='-', c = 'blue', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y1,color='blue')
    
    plt.plot(x, y2, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y2,color='red')

    plt.plot(x, y3, ls='-', c = 'green', alpha = 0.5, linewidth = 2.0, linestyle='-') 
    plt.scatter(x,y3,color='green')

    x_axis = [0, 510]

    plt.xlim(0, 510)
    # plt.xticks( np.arange(min(y), max(y), 25) )
    # plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
    plt.xlabel(u"Amostras")
    plt.ylabel(u"Perímetro")

    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]

    plt.legend(custom_lines, ['Dumper', 'Clamp', 'Cable'])
    # plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
    # Grid
    plt.grid(True)
    plt.show()