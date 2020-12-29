# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:45:59 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
    
n_estimators: in general the more trees the less likely the algorithm is to overfit. So try increasing this. The lower this number, the closer the model is to a decision tree, with a restricted feature set.
max_features: try reducing this number (try 30-50% of the number of features). This determines how many features each tree is randomly assigned. The smaller, the less likely to overfit, but too small will start to introduce under fitting.
max_depth: Experiment with this. This will reduce the complexity of the learned models, lowering over fitting risk. Try starting small, say 5-10, and increasing you get the best result.
min_samples_leaf: Try setting this to values greater than one. This has a similar effect to the max_depth parameter, it means the branch will stop splitting once the leaves have that number of samples each

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
n_estimators_ = 500
max_features_ = 10
max_depth_ = 10
min_samples_leaf_ = 10
with_PCA_=True

print('\nRandom Forest Model \n')
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
    
