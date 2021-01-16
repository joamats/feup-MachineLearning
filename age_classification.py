# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:52:46 2021

@author: User
"""

from data_load import getDataset, getTestDataset
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics, displayMetrics
import numpy as np
from sklearn import svm
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

withGenderSeparation = False;

def ageProcessing (y):

    yProcessed = np.zeros(y.shape)
    

    for i in range(y.shape[0]):
       
        if y[i] <= 16 :
            yProcessed[i] = 0
        elif ((y[i]>16 and y[i]<=25) == True):
            yProcessed[i] = 1
        else: 
            yProcessed[i]=2;
            
    return yProcessed


metrics_tr = []
metrics_val = []
language = ''
numMetrics = 5 # Accuracy, F1-score, precision score, recall score, balanced accuracy
numValues = 2   # Mean, Standard Deviation 
mode= 'SubjectIndependent'
if (withGenderSeparation):
 genders =  ['Male', 'Female']
 
 
# cross validation for train
 for k, gender in enumerate(genders): 
    for number in range(10):
             
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val = getDataset(number, language, mode, gender)
        x_ts, y_ts = getTestDataset(language, mode, gender)
    
                
        ageLabeltr = ageProcessing(y_tr[:,2])
        ageLabelval= ageProcessing(y_val[:,2])
        
         
        smote= SMOTE(random_state=2)
        x_training, age_training = smote.fit_resample(x_tr, ageLabeltr)  
        
      
       
        # # Train SVC
      
        C_ = 0.1
        kernel_ = 'rbf'
        gamma_ = 1.8
        model = svm.SVC(random_state=0, C=C_, kernel=kernel_, gamma=gamma_)      

        model.fit(x_training, age_training)
       
       
      # Assess *this* model
        metrics_tr.append(getMetrics(model, x_training, age_training, None, 'Multiclass'))
        metrics_val.append(getMetrics(model, x_val, ageLabelval, None, 'Multiclass'))

    #get mean and std for each metric
    sMetrics_tr = getGeneralMetrics(metrics_tr, numMetrics)
    sMetrics_val = getGeneralMetrics(metrics_val, numMetrics)

    print('\nGender:', gender,'\n')
    print('Training Set')
    displayGeneralMetrics(sMetrics_tr, 'Multiclass')
    print('\n')
    print('Validation Set')
    displayGeneralMetrics(sMetrics_val, 'Multiclass')
    print('\n')      
    print('Testing Set')

    #Final model and test
    x_TR = np.concatenate((x_tr,x_val), axis=0)
    y_TR = np.concatenate((ageLabeltr, ageLabelval), axis=0)
    x_TR, y_TR = smote.fit_resample(x_TR, y_TR)
    model.fit(x_TR, y_TR)
    metrics = getMetrics(model, x_ts, y_ts, None, 'Multiclass')
    displayMetrics(metrics,'Multiclass')
    
else:
  for number in range(10):
             
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val = getDataset(number, '', mode)       
        x_ts, y_ts = getTestDataset('', mode)
             
        ageLabeltr = ageProcessing(y_tr[:,2])
        ageLabelval= ageProcessing(y_val[:,2])        
        #y_ts = ageProcessing(y_ts[:,2])
        
        smote= SMOTE(random_state=2)
        x_training, age_training = smote.fit_resample(x_tr, ageLabeltr)  
        x_validation, age_validation = smote.fit_resample(x_val, ageLabelval)     
      
        C_ = 0.1
        kernel_ = 'rbf'
        gamma_ = 1.8
        model = svm.SVC(random_state=0, C=C_, kernel=kernel_, gamma=gamma_)      

        model.fit(x_training, age_training)
       
        #Assess *this* model
        metrics_tr.append(getMetrics(model, x_training, age_training, None, 'Multiclass'))
        metrics_val.append(getMetrics(model, x_val, ageLabelval , None, 'Multiclass'))

# #get mean and std for each metric
sMetrics_tr = getGeneralMetrics(metrics_tr, numMetrics)
sMetrics_val = getGeneralMetrics(metrics_val, numMetrics)

print('Training Set')
displayGeneralMetrics(sMetrics_tr, 'Multiclass')
print('\n')
print('Validation Set')
displayGeneralMetrics(sMetrics_val, 'Multiclass')
print('\n')      
print('Testing Set')

#Final model and test
x_TR = np.concatenate((x_tr,x_val), axis=0)
y_TR = np.concatenate((ageLabeltr, ageLabelval), axis=0)
x_TR, y_TR = smote.fit_resample(x_TR, y_TR)
model.fit(x_TR, y_TR)
y_test = ageProcessing(y_ts[:,2])
metrics = getMetrics(model, x_ts, y_test, None, 'Multiclass')
displayMetrics(metrics,'Multiclass')

# Comparison of English / Native utterances
lanEnglish_TS = y_ts[:,4] #0 for English; 1 for Native
x_TS_english = x_ts[np.where(lanEnglish_TS == 0)]
y_TS_english = y_ts[np.where(lanEnglish_TS == 0)]
y_ts_english = ageProcessing(y_TS_english[:,2])

x_TS_native = x_ts[np.where(lanEnglish_TS == 1)]
y_TS_native = y_ts[np.where(lanEnglish_TS == 1)]
y_ts_native = ageProcessing(y_TS_native[:,2])
metrics = getMetrics(model, x_TS_native, y_ts_native, None, 'Multiclass')
displayMetrics(metrics,'Multiclass')



#Confusion matrix

y_TS = y_test.reshape(y_test.shape[0],1).astype(float)
pred_TS_classes = model.predict(x_ts)       
c = confusion_matrix(y_TS, np.round(pred_TS_classes), np.unique(y_TS))
df_cm = pd.DataFrame(c, [s for s in np.unique(y_TS).astype(int)], [s for s in np.unique(y_TS).astype(int)])
sn.set(font_scale=1.1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap="YlGnBu") # font size


plt.title("Confusion Matrix of final Age Classification")
plt.xlabel("Predicted Classes", fontsize=10)
plt.ylabel("Actual Classes", fontsize=10)




