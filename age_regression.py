# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:14:08 2021

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""
import numpy as np
import pandas as pd
import seaborn as sn
from timeit import default_timer as timer
from data_load import getDataset, getTestDataset
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from evaluation_metrics import getGeneralMetrics, displayGeneralMetrics, displayMetrics
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix

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

def ageMLPRegressor():
    reg = MLPRegressor(random_state=2, activation='relu', 
                         hidden_layer_sizes=(500,500),
                         max_iter=500, 
                         early_stopping=True, validation_fraction=0.1,
                         alpha=.1)
    
    return reg


def ageRidgeRegression():
    sample_weight = compute_sample_weight("balanced", y_tr)
    reg = Ridge(alpha = .01)
    return reg, sample_weight
    

#%% Cross Validation for Train
start = timer()

metrics_tr = []
metrics_val = []

# cross validation for train
for number in range(10):

    language = ''
    mode = 'SubjectIndependent'
    x_tr, y_tr, x_val, y_val,  = getDataset(number, language, mode)
    x_TS, y_TS = getTestDataset(language, mode)
    
    lanEnglish_tr = y_tr[:,4]
    lanEnglish_val = y_val[:,4]
    lanEnglish_TS = y_TS[:,4] #0 for English; 1 for Native
    
    y_tr = y_tr[:,2]
    y_val = y_val[:,2]
    y_TS = y_TS[:,2]

    # ros = RandomOverSampler(random_state=42)
    # x_tr, y_tr = ros.fit_resample(x_tr, y_tr)

    #reg = ageMLPRegressor()
    reg, sample_weight = ageRidgeRegression()

    reg.fit(x_tr,y_tr, sample_weight)
    
    print(reg.score(x_tr, y_tr))
    preds_tr = reg.predict(x_tr)
    preds_val = reg.predict(x_val)

    pred_tr_classes = ageProcessing(preds_tr)
    pred_val_classes = ageProcessing(preds_val)
    
    y_tr_classes = ageProcessing(y_tr).reshape(y_tr.shape[0],1).astype(float)
    y_val_classes = ageProcessing(y_val).reshape(y_val.shape[0],1).astype(float)
    
    metrics_tr.append([accuracy_score(y_tr_classes, pred_tr_classes),
                       f1_score(y_tr_classes, pred_tr_classes, average='weighted'),
                       precision_score(y_tr_classes, pred_tr_classes, average='weighted'),
                       recall_score(y_tr_classes, pred_tr_classes, average='weighted')]) #get recall score
    
    metrics_val.append([accuracy_score(y_val_classes, pred_val_classes), 
                        f1_score(y_val_classes, pred_val_classes, average='weighted'),
                        precision_score(y_val_classes, pred_val_classes, average='weighted'),
                        recall_score(y_val_classes, pred_val_classes, average='weighted')])
        
#get mean and std for each metric
sMetrics_tr = getGeneralMetrics(metrics_tr, 4)
sMetrics_val = getGeneralMetrics(metrics_val, 4)        
print('\nLanguage:', language,'\n')
print('Training Set')
displayGeneralMetrics(sMetrics_tr)
print('\n')
print('Validation Set')
displayGeneralMetrics(sMetrics_val)
print('\n')

#%% Final train, before test

x_TR = np.concatenate((x_tr,x_val), axis=0)
y_TR = np.concatenate((y_tr,y_val), axis=0)
lanEnglish_TR = np.concatenate((lanEnglish_tr,lanEnglish_val), axis=0)

sample_weight = compute_sample_weight("balanced", y_TR)
reg.fit(x_TR, y_TR)#, sample_weight)

pred_TR = reg.predict(x_TR)

pred_TR_classes = ageProcessing(pred_TR)
y_TR_classes = ageProcessing(y_TR).reshape(y_TR.shape[0],1).astype(float)

print('\nComplete Train Set')
metrics = ([accuracy_score(y_TR_classes, pred_TR_classes),
            f1_score(y_TR_classes, pred_TR_classes, average='weighted'),
            precision_score(y_TR_classes, pred_TR_classes, average='weighted'),
            recall_score(y_TR_classes, pred_TR_classes, average='weighted')]) #get recall score

displayMetrics(metrics)

# Comparison of English / Native utterances
x_TR_english = x_TR[np.where(lanEnglish_TR == 0)]
y_TR_english = y_TR[np.where(lanEnglish_TR == 0)]

x_TR_native = x_TR[np.where(lanEnglish_TR == 1)]
y_TR_native = y_TR[np.where(lanEnglish_TR == 1)]

# get predictions 
pred_TR_english = reg.predict(x_TR_english)
pred_TR_native = reg.predict(x_TR_native)

# conversion to classes
pred_TR_classes_english = ageProcessing(pred_TR_english)
pred_TR_classes_native = ageProcessing(pred_TR_native)
y_TR_classes_english = ageProcessing(y_TR_english).reshape(y_TR_english.shape[0],1).astype(float)
y_TR_classes_native = ageProcessing(y_TR_native).reshape(y_TR_native.shape[0],1).astype(float)


print('\nEnglish Results:')
metrics = ([accuracy_score(y_TR_classes_english, pred_TR_classes_english),
            f1_score(y_TR_classes_english, pred_TR_classes_english, average='weighted'),
            precision_score(y_TR_classes_english, pred_TR_classes_english, average='weighted'),
            recall_score(y_TR_classes_english, pred_TR_classes_english, average='weighted')]) #get recall score
    
displayMetrics(metrics)

print('\nNative Results:')
metrics = ([accuracy_score(y_TR_classes_native, pred_TR_classes_native),
            f1_score(y_TR_classes_native, pred_TR_classes_native, average='weighted'),
            precision_score(y_TR_classes_native, pred_TR_classes_native, average='weighted'),
            recall_score(y_TR_classes_native, pred_TR_classes_native, average='weighted')]) #get recall score
    
displayMetrics(metrics)


#%% Test Model

pred_TS = reg.predict(x_TS)

pred_TS_classes = ageProcessing(pred_TS)
y_TS_classes = ageProcessing(y_TS).reshape(y_TS.shape[0],1).astype(float)

print('\nTest Set')
metrics = ([accuracy_score(y_TS_classes, pred_TS_classes),
            f1_score(y_TS_classes, pred_TS_classes, average='weighted'),
            precision_score(y_TS_classes, pred_TS_classes, average='weighted'),
            recall_score(y_TS_classes, pred_TS_classes, average='weighted')]) #get recall score

displayMetrics(metrics)

# Comparison of English / Native utterances
x_TS_english = x_TS[np.where(lanEnglish_TS == 0)]
y_TS_english = y_TS[np.where(lanEnglish_TS == 0)]

x_TS_native = x_TS[np.where(lanEnglish_TS == 1)]
y_TS_native = y_TS[np.where(lanEnglish_TS == 1)]

# get predictions 
pred_TS_english = reg.predict(x_TS_english)
pred_TS_native = reg.predict(x_TS_native)

# conversion to classes
pred_TS_classes_english = ageProcessing(pred_TS_english)
pred_TS_classes_native = ageProcessing(pred_TS_native)
y_TS_classes_english = ageProcessing(y_TS_english).reshape(y_TS_english.shape[0],1).astype(float)
y_TS_classes_native = ageProcessing(y_TS_native).reshape(y_TS_native.shape[0],1).astype(float)

print('\nEnglish Results:')
metrics = ([accuracy_score(y_TS_classes_english, pred_TS_classes_english),
            f1_score(y_TS_classes_english, pred_TS_classes_english, average='weighted'),
            precision_score(y_TS_classes_english, pred_TS_classes_english, average='weighted'),
            recall_score(y_TS_classes_english, pred_TS_classes_english, average='weighted')]) #get recall score
    
displayMetrics(metrics)

print('\nNative Results:')
metrics = ([accuracy_score(y_TS_classes_native, pred_TS_classes_native),
            f1_score(y_TS_classes_native, pred_TS_classes_native, average='weighted'),
            precision_score(y_TS_classes_native, pred_TS_classes_native, average='weighted'),
            recall_score(y_TS_classes_native, pred_TS_classes_native, average='weighted')]) #get recall score
    
displayMetrics(metrics)

#%%
# Confusion Matrix to assess the regression
c = confusion_matrix(y_TS, np.round(pred_TS), np.unique(y_TS))
df_cm = pd.DataFrame(c, [s for s in np.unique(y_TS).astype(int)], [s for s in np.unique(y_TS).astype(int)])
sn.set(font_scale=1.1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap="YlGnBu") # font size

plt.title("Confusion Matrix of Age Ridge Regression")
plt.xlabel("Predicted Ages", fontsize=10)
plt.ylabel("Actual Ages", fontsize=10)
 
plt.show()

# Regression Score
print(reg.score(x_TS, y_TS))

# Confusion Matrix to assess the final classification
c = confusion_matrix(y_TS_classes, np.round(pred_TS_classes), np.unique(y_TS_classes))
df_cm = pd.DataFrame(c, [s for s in np.unique(y_TS_classes).astype(int)], [s for s in np.unique(y_TS_classes).astype(int)])
sn.set(font_scale=1.1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, cmap="YlGnBu") # font size

plt.title("Confusion Matrix of final Age Classification, with Costs")
plt.xlabel("Predicted Classes", fontsize=10)
plt.ylabel("Actual Classes", fontsize=10)
 
plt.show()

#%% Obtain Histogram showing imbalance data

plt.hist(y_TR, bins=np.unique(y_TR).shape[0]+1, color='b')
plt.axvline(17, color='r', linestyle='dashed', linewidth=2)
plt.axvline(25, color='r', linestyle='dashed', linewidth=2)
plt.xlabel("Ages")
plt.ylabel("Counts")
plt.title("Histogram of Training Data Distribution")
plt.show()
end = timer()
print(f'{end-start:.3f}', 's')
