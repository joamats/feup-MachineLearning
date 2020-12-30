# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

import numpy as np
from timeit import default_timer as timer
from sklearn import svm
from data_load import getDataset
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

start = timer()

#%% SVM

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 5  # Accuracy, F1-score, Confusion Matrix
numValues = 2   # Mean, Standard Deviation   

sMetrics_tr = np.zeros((numLanguages, numMetrics, numValues))
sMetrics_val = np.zeros((numLanguages, numMetrics, numValues))

# Define Parameters for Train

probability_ = True
C_ = 100
kernel_ = 'rbf'
gamma_ = 0.01
with_PCA_= True
with_ANOVA_ = True
k_features_ = 23

print('SVM Model')
print('C =', C_)
print('kernel =', kernel_)
print('gamma =', gamma_)
print('With PCA:', with_PCA_)
print('With ANOVA:', with_ANOVA_)
if with_ANOVA_:
    print('k features:', k_features_)

for k, language in enumerate(languages):
    
    metrics_tr = []
    metrics_val = []
    #best_ = []

    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language, with_PCA = with_PCA_)
        
        # SVM model
        svc = svm.SVC(probability=probability_, random_state=0, C=C_, kernel=kernel_, gamma=gamma_)
        
        if with_ANOVA_:
            # ANOVA filter, take 3 best features
            anova_filter = SelectKBest(f_classif, k=k_features_)
        
            # Make Pipeline
            model = make_pipeline(anova_filter, svc)
        else:
            model = svc
    
        model.fit(x_tr, y_tr)
                        
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
    
end = timer()
print(f'{end-start:.3f}', 's')
