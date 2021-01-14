# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:14:33 2021

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""
import numpy as np
from timeit import default_timer as timer
from data_load import getDataset, getTestDataset
from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics, displayMetrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from feature_selection import plot_score_features
from sklearn import preprocessing as pp

#%% SVM models for Gender
def genderSVM():
    
    # SVM Parameters
    probability_ = True
    C_ = 100
    kernel_ = 'linear'
    gamma_ = 0.01
    with_PCA_= False
    PCA_variability_ = 0.95
    with_ANOVA_ = True
    k_features_ = 3
    
    # Create Model
    svc = svm.SVC(probability=probability_,
                  random_state=42, 
                  C=C_, 
                  kernel=kernel_, 
                  gamma=gamma_)

    # Pipelines
    if with_PCA_:
        #Create PCA to redimention features to have X % variability
        pca = PCA(n_components=PCA_variability_, random_state=42)
    
        model = make_pipeline(pca, svc)
        
    elif with_ANOVA_:
    
        #ANOVA filter, take best features
        anova_filter = SelectKBest(f_classif, k=k_features_)
    
        # Make Pipeline
        model = make_pipeline(anova_filter, svc)
    else: 
        model = svc
    
    return model

#%% Logistic Regression model for Gender

def genderLogReg(x_tr, x_val, x_):
    #normalizing the features
    scaler = pp.StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr) 
    
    scaler.fit(x_val)
    x_val = scaler.transform(x_val)
    
    scaler.fit(x_TS)
    x_TS = scaler.transform(x_TS)
    
    # Train Logistics Regression
    model = LogisticRegressionCV(Cs= 60, solver='newton-cg', random_state=0)

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

    y_tr = y_tr[:,1]
    y_val = y_val[:,1]
    y_TS = y_TS[:,1] #age

    model = genderSVM()
    model.fit(x_tr, y_tr)

    # Assess *this* model
    metrics_tr.append(getMetrics(model, x_tr, y_tr, 'withProbs'))
    metrics_val.append(getMetrics(model, x_val, y_val, 'withProbs'))


sMetrics_tr = getGeneralMetrics(metrics_tr, 5)
sMetrics_val = getGeneralMetrics(metrics_val, 5)

print('Cross-Validation\n')
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

model.fit(x_TR, y_TR)

print('\nComplete Train Set')
metrics = getMetrics(model, x_TR, y_TR, 'withProbs')
displayMetrics(metrics)

# Comparison of English / Native utterances
x_TR_english = x_TR[np.where(lanEnglish_TR == 0)]
y_TR_english = y_TR[np.where(lanEnglish_TR == 0)]

x_TR_native = x_TR[np.where(lanEnglish_TR == 1)]
y_TR_native = y_TR[np.where(lanEnglish_TR == 1)]

print('\nEnglish Results:')
metrics = getMetrics(model, x_TR_english, y_TR_english, 'withProbs')
displayMetrics(metrics)

print('\nNative Results:')
metrics = getMetrics(model, x_TR_native, y_TR_native, 'withProbs')
displayMetrics(metrics)

plot_score_features(x_TR, y_TR)

#%% Test Model

print('\nTest Set')
metrics = getMetrics(model, x_TS, y_TS, 'withProbs')
displayMetrics(metrics)

# Comparison of English / Native utterances
x_TS_english = x_TS[np.where(lanEnglish_TS == 0)]
y_TS_english = y_TS[np.where(lanEnglish_TS == 0)]

x_TS_native = x_TS[np.where(lanEnglish_TS == 1)]
y_TS_native = y_TS[np.where(lanEnglish_TS == 1)]

print('\nEnglish Results:')
metrics = getMetrics(model, x_TS_english, y_TS_english, 'withProbs')
displayMetrics(metrics)

print('\nNative Results:')
metrics = getMetrics(model, x_TS_native, y_TS_native, 'withProbs')
displayMetrics(metrics)


end = timer()
print(f'{end-start:.3f}', 's')
