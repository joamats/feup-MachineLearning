# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:46:22 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from sklearn.linear_model import LogisticRegressionCV
from data_load import getDataset
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
import numpy as np
from sklearn import preprocessing as pp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

metrics_tr = []
metrics_val = []
languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 4 # Accuracy, F1-score, precision score, recall score, roc_auc
numValues = 2   # Mean, Standard Deviation 

for k, language in enumerate(languages):
    
    
    for number in range(10):
             
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val = getDataset(number, language)
        
        #getting only the gender labels
        y_tr = y_tr[:,0] 
        y_val = y_val[:,0]
        #normalizing the features
        scaler = pp.StandardScaler()
        scaler.fit(x_tr)
        x_train_normalized = scaler.transform(x_tr) 
        scaler.fit(x_val)
        x_val_normalized = scaler.transform(x_val)
        
   
        # Train Logistics Regression
        model = LogisticRegressionCV(Cs= 60, solver='newton-cg', random_state=0)
    
        model.fit(x_train_normalized, y_tr)
       
        # Assess *this* model
        metrics_tr.append(getMetrics(model, x_train_normalized , y_tr))
        metrics_val.append(getMetrics(model, x_val_normalized, y_val))

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
    
