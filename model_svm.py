# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

import numpy as np
from sklearn import svm
from data_load import getDataset
from evaluation_metrics import getMetrics, displayMetrics

#%% Load Datasets

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 3  # Accuracy, F1-score, Confusion Matrix
numValues = 2   # Mean, Standard Deviation   

allMetrics = np.zeros((numLanguages, numMetrics, numValues))

for k, language in enumerate(languages):
    
    metrics = []
    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language)
        
        # Train SVM
        model = svm.SVC(kernel = "linear")
        model.fit(x_tr, y_tr)
        
        # Assess *this* model
        metrics.append(getMetrics(model, x_val, y_val))
        
    # Mean and Std for Accuracy    
    allMetrics[k, 0, 0] = np.mean([r[0] for r in metrics])
    allMetrics[k, 0, 1] = np.std([r[0] for r in metrics])
    
    # Mean and Std for F1-score
    allMetrics[k, 1, 0] = np.mean([r[1] for r in metrics])
    allMetrics[k, 1, 1] = np.std([r[1] for r in metrics])
    
    print('Language:', language)
    print('Accuracy =', f'{allMetrics[k,0,0]:.3f}', '+/-',  f'{allMetrics[k,0,1]:.3f}')
    print('F1-Score =', f'{allMetrics[k,1,0]:.3f}', '+/-', f'{allMetrics[k,1,1]:.3f}\n')

