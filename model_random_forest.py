# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:45:59 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from data_load import getDataset

#%% Load Datasets

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 6  
numValues = 2   # Mean, Standard Deviation   

sMetrics_tr = np.zeros((numLanguages, numMetrics, numValues))
sMetrics_val = np.zeros((numLanguages, numMetrics, numValues))


# Define Parameters for Train

n_estimators_ = 200
max_features_ = 15
max_depth_ = 7
min_samples_leaf_ = 2
with_PCA_=True

print('Random Forest Model \n')
print('PCA selection: ', with_PCA_)
print('n_estimators =', n_estimators_)
print('max_features =', max_features_)
print('max_depth =', max_depth_)
print('min_samples_leaf =', min_samples_leaf_)

for k, language in enumerate(languages):
    
    metrics_tr = []
    metrics_val = []
    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language, with_PCA=with_PCA_)
        
        # Train SVM
        model = RandomForestClassifier(n_estimators=n_estimators_, max_features=max_features_, max_depth= max_depth_, min_samples_leaf=min_samples_leaf_, random_state=0).fit(x_tr, y_tr)
        
        # Assess *this* model
        metrics_tr.append(getMetrics(model, x_tr, y_tr, 'withProbs'))
        metrics_val.append(getMetrics(model, x_val, y_val, 'withProbs'))

    #get mean and std for each metric
    sMetrics_tr = getGeneralMetrics(metrics_tr, numMetrics)
    sMetrics_val = getGeneralMetrics(metrics_val, numMetrics)

    print('\nLanguage:', language,'\n')
    print('Training Set')
    displayGeneralMetrics(sMetrics_tr)
    print('\n')
    print('Validation Set')
    displayGeneralMetrics(sMetrics_val)
    print('\n')
    
