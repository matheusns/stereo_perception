#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn import preprocessing, model_selection, utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys
from matplotlib.lines import Line2D
import matplotlib as mat
import matplotlib.pyplot as plt

try:
    # the file name is passed as argument when you gonna run the script
    file_name = str(sys.argv[1])

    # The number os neighbors used in the KNN algorithm. Is passed as argument, as well.
    k = int(sys.argv[2])
except:
    file_name = "dataset.txt"
    k = 5
    
df = pd.read_csv(file_name)
df.drop(['id'], 1, inplace=True)

accuracy_array = []
accuracy_min_array = []
accuracy_std_array = []
train_accuracy = []
test_accuracy = []

# how many times the classification will run
test_times = 500
samples = 0
axis_x = []

for i in range(1,test_times):
    random_data = utils.shuffle(df)
    samples += 1
    knn_stardard_pipe = make_pipeline(StandardScaler(),
                                    KNeighborsClassifier(n_neighbors=k) ) 

    knn_minMax_pipe = make_pipeline(MinMaxScaler(),
                                    KNeighborsClassifier(n_neighbors=k))
    
    clf = KNeighborsClassifier(n_neighbors=k)

    # features = ['area','aspects','solidities','extents','perimeters','labels'] 

    dropped = ['labels'] 

    X = np.array(random_data.drop(dropped,1))
    y = np.array(random_data['labels'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)


    clf.fit(X_train,y_train)
    knn_minMax_pipe.fit(X_train,y_train)
    knn_stardard_pipe.fit(X_train,y_train)

    accuracy_knn = clf.score(X_test,y_test)
    accuracy_knn_std = knn_stardard_pipe.score(X_test,y_test)
    accuracy_knn_minMax = knn_minMax_pipe.score(X_test,y_test)
    
    print 'KNN', accuracy_knn
    print 'Standarized KNN', accuracy_knn_std
    print 'MinMax KNN', accuracy_knn_minMax

    # predicted = clf.predict(testing)

    accuracy_array.append(accuracy_knn)
    accuracy_min_array.append(accuracy_knn_std)
    accuracy_std_array.append(accuracy_knn_minMax)
    axis_x.append(samples)

    y_pred = knn_stardard_pipe.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred) # Calulate Confusion matrix for test set.

    cm = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print cm

    train_accuracy.append(knn_stardard_pipe.score(X_train, y_train))
    
    #Compute accuracy on the test set
    test_accuracy.append(knn_stardard_pipe.score(X_test, y_test) )
#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(axis_x, test_accuracy, label='Testing Accuracy')
plt.plot(axis_x, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()



print ''
print '====================================================='
print 'media KNN', np.mean(accuracy_array)
print 'mediana KNN', np.median(accuracy_array)
print '====================================================='
print ''
print '====================================================='
print 'media MinMax KNN', np.mean(accuracy_min_array)
print 'mediana MinMax KNN', np.median(accuracy_min_array)
print '====================================================='
print ''
print '====================================================='
print 'media Standarized KNN', np.mean(accuracy_std_array)
print 'mediana Standarized KNN', np.median(accuracy_std_array)
print '====================================================='
print ''

# -------------------------------------------------
# MatplotLib Parameters
# -------------------------------------------------
fontsize = 18
mat.rc('legend', fontsize=fontsize, handlelength=3)
mat.rc('axes', titlesize=fontsize)
mat.rc('axes', labelsize=25)
mat.rc('xtick', labelsize=fontsize)
mat.rc('ytick', labelsize=fontsize)
# mat.rc('text', usetex=True)
mat.rc('font', size=fontsize, family='serif', style='normal', variant='normal',stretch='normal', weight='normal')

plt.figure(1)
plt.plot(axis_x, accuracy_array, ls='-', c = 'orange', alpha = 0.5, linewidth = 2.0, linestyle='-', label='knn') 
plt.scatter(axis_x,accuracy_array,color='orange')

plt.plot(axis_x, accuracy_min_array, ls='-', c = 'red', alpha = 0.5, linewidth = 2.0, linestyle='-', label='MinMax') 
plt.scatter(axis_x,accuracy_min_array,color='red')

plt.plot(axis_x, accuracy_std_array, ls='-', c = 'black', alpha = 0.5, linewidth = 2.0, linestyle='-', label='Standarized') 
plt.scatter(axis_x,accuracy_std_array,color='black')

y = [0,1]
x_axis = [0, test_times]

plt.xlim(0, test_times+1)
plt.xticks( np.arange(min(x_axis), max(x_axis), 25) )
# plt.yticks( np.arange(min(x_axis), max(x_axis), 0.02) )
plt.xlabel(u"Iterações")
plt.ylabel(u"Precisão de acerto")

custom_lines = [Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='black', lw=4)]

plt.legend(custom_lines, ['Knn', 'MinMax', 'Standarized'])
# plt.plot( amortecedores_area, 'b-o',label="amortecedor" )
# Grid
plt.grid(True)
# plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()