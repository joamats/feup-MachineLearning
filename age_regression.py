# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:14:08 2021

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""
import numpy as np
from data_load import getDataset
from sklearn.linear_model import LinearRegression, PoissonRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics

def ageProcessing (y):

    yProcessed = np.zeros(y.shape)

    for i in range(y.shape[0]):
        if y[i] <= 17 :
            yProcessed[i] = 0
        elif ((y[i]>17 and y[i]<30) == True):
            yProcessed[i] = 1
        else: 
            yProcessed[i]=2;
            
    return yProcessed


#%% Get Dataset


languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 4
numValues = 2   # Mean, Standard Deviation 

for k, language in enumerate(languages):    
    
    for number in range(2):

        x_tr, y_tr, x_val, y_val = getDataset(number, language)
        y_tr = y_tr[:,1]
        y_val = y_val[:,1]

        #%% RandomOverSample
        ros = RandomOverSampler(random_state=42, sampling_strategy='minority') 
        x_tr, y_tr = ros.fit_resample(x_tr, y_tr)
        
        #%% Train Linear Regression
        
        reg = MLPRegressor(random_state=2, activation='relu', 
                           hidden_layer_sizes=(100,3),
                            max_iter=1000, 
                            early_stopping=True, validation_fraction=0.1,
                            alpha=.01)
                
        reg.fit(x_tr,y_tr)
        preds_tr = reg.predict(x_tr)
        preds_val = reg.predict(x_val)
        
        pred_tr_classes = ageProcessing(preds_tr)
        pred_val_classes = ageProcessing(preds_val)
        #%%
        y_tr_classes = ageProcessing(y_tr).reshape(y_tr.shape[0],1).astype(float)
        y_val_classes = ageProcessing(y_val).reshape(y_val.shape[0],1).astype(float)
        
        metrics_tr = []
        metrics_tr.append(accuracy_score(y_tr_classes, pred_tr_classes))
        metrics_tr.append(f1_score(y_tr_classes, pred_tr_classes, average='weighted')) #get f1 score
        metrics_tr.append(precision_score(y_tr_classes, pred_tr_classes, average='weighted')) #get precision score
        metrics_tr.append(recall_score(y_tr_classes, pred_tr_classes, average='weighted')) #get recall score
        
        metrics_val = []
        metrics_val.append(accuracy_score(y_val_classes, pred_val_classes))
        metrics_val.append(f1_score(y_val_classes, pred_val_classes, average='weighted')) #get f1 score
        metrics_val.append(precision_score(y_val_classes, pred_val_classes, average='weighted')) #get precision score
        metrics_val.append(recall_score(y_val_classes, pred_val_classes, average='weighted')) #get recall score
         
        
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
    
        
        




