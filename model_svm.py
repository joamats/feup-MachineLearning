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
from evaluation_metrics import getMetrics

#%% Load Datasets

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 3  # Accuracy, F1-score, Confusion Matrix
numValues = 2   # Mean, Standard Deviation   

sMetrics_tr = np.zeros((numLanguages, numMetrics, numValues))
sMetrics_val = np.zeros((numLanguages, numMetrics, numValues))

for k, language in enumerate(languages):
    
    metrics_tr = []
    metrics_val = []
    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language)
        
        # Train SVM
        model = svm.SVC(kernel = "linear")
        model.fit(x_tr, y_tr)
        
        # Assess *this* model
        metrics_tr.append(getMetrics(model, x_tr, y_tr))
        metrics_val.append(getMetrics(model, x_val, y_val))
        
    # Mean and Std for Accuracy    
    sMetrics_tr[k, 0, 0] = np.mean([r[0] for r in metrics_tr])
    sMetrics_tr[k, 0, 1] = np.std([r[0] for r in metrics_tr])
    
    # Mean and Std for F1-score
    sMetrics_tr[k, 1, 0] = np.mean([r[1] for r in metrics_tr])
    sMetrics_tr[k, 1, 1] = np.std([r[1] for r in metrics_tr])
    
    # Mean and Std for Accuracy    
    sMetrics_val[k, 0, 0] = np.mean([r[0] for r in metrics_val])
    sMetrics_val[k, 0, 1] = np.std([r[0] for r in metrics_val])
    
    # Mean and Std for F1-score
    sMetrics_val[k, 1, 0] = np.mean([r[1] for r in metrics_val])
    sMetrics_val[k, 1, 1] = np.std([r[1] for r in metrics_val])
    
    print('\nLanguage:', language,'\n')
    print('Training Set')
    print('Accuracy =', f'{sMetrics_tr[k,0,0]:.3f}', '+/-',  f'{sMetrics_tr[k,0,1]:.3f}')
    print('F1-Score =', f'{sMetrics_tr[k,1,0]:.3f}', '+/-', f'{sMetrics_tr[k,1,1]:.3f}\n')
    print('Validation Set')
    print('Accuracy =', f'{sMetrics_val[k,0,0]:.3f}', '+/-',  f'{sMetrics_val[k,0,1]:.3f}')
    print('F1-Score =', f'{sMetrics_val[k,1,0]:.3f}', '+/-', f'{sMetrics_val[k,1,1]:.3f}\n')

