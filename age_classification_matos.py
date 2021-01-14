# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:35:11 2021

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
"""

import numpy as np
from data_load import getDataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn import svm
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
from sklearn.metrics import classification_report

def ageProcessing (y):

    yProcessed = np.zeros(y.shape)
    
    for i in range(y.shape[0]):
        if y[i] >= 25:
            yProcessed[i] = 1

        else: 
            yProcessed[i]=0;
            
    return yProcessed

#%% 

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
        
        #getting only the age labels
        y_tr = y_tr[:,1] 
        y_val = y_val[:,1]
        
        y_tr_classes = ageProcessing(y_tr)
        y_val_classes = ageProcessing(y_val)

        smote= SMOTE(random_state=42)
        x_tr, y_tr_classes = smote.fit_resample(x_tr, y_tr_classes)
 
        C_ = 1000
        kernel_ = 'rbf'
        gamma_ = .1
        model = svm.SVC(probability=True,
                        class_weight='balanced',
                        random_state=42, 
                        C=C_, 
                        kernel=kernel_, 
                        gamma=gamma_)
        model.fit(x_tr, y_tr_classes)
        
        pred_tr = model.predict(x_tr)
        pred_val = model.predict(x_val)
       
        print('Train\n', classification_report(y_tr_classes, pred_tr))
        print('Validation\n', classification_report(y_val_classes, pred_val))
        

    