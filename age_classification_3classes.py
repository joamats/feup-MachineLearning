# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:52:46 2021

@author: User
"""
from sklearn.linear_model import LogisticRegressionCV
from data_load import getDataset
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
import numpy as np
from sklearn import svm
from sklearn import preprocessing as pp
from imblearn.over_sampling import SMOTE

def ageProcessing (y):

    yProcessed = np.zeros(y.shape)
    

    for i in range(y.shape[0]):
        if y[i] <= 17 :
            yProcessed[i] = 0
        elif ((y[i]>17 and y[i]<60) == True):
            yProcessed[i] = 1
        else: 
            yProcessed[i]=2;
            
    return yProcessed

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
        y_tr = y_tr[:,1] 
        y_val = y_val[:,1]
        
        y_train = ageProcessing(y_tr)
        y_validation = ageProcessing(y_val)
        
        
        smote= SMOTE(random_state=2)
        x_tr, y_train = smote.fit_resample(x_tr, y_train)
        
        #normalizing the features
        # scaler = pp.StandardScaler()
        # scaler.fit(x_tr)
        # x_train_normalized = scaler.transform(x_tr) 
        # scaler.fit(x_val)
        # x_val_normalized = scaler.transform(x_val)
        
   
 
        # Train Logistics Regression
        model = LogisticRegressionCV(Cs= 25, solver='newton-cg', multi_class='multinomial', random_state=0)
        C_ = 100
        kernel_ = 'rbf'
        gamma_ = 0.01
        #model = svm.SVC(random_state=0, C=C_, kernel=kernel_, gamma=gamma_)
        model.fit(x_tr, y_train)
       
        # Assess *this* model
        preds = getMetrics(model, x_tr, y_train)
        predsval = getMetrics(model, x_val, y_validation)
        metrics_tr.append(a)
        metrics_val.append(b)
        

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
    
    
