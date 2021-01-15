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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#%% SVM models for Gender

def genderSVM():
    
    # SVM Parameters
    probability_ = True
    C_ = 100
    kernel_ = 'rbf'
    gamma_ = 0.01
    with_PCA_= True
    PCA_variability_ = 0.95
    with_ANOVA_ = False
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

def genderLogReg():
    #normalizing the features
    scaler = pp.StandardScaler()
    
    # Train Logistics Regression
    logReg = LogisticRegressionCV(Cs= 50, solver='newton-cg', random_state=42)
    
    model = make_pipeline(scaler, logReg)
    
    with_PCA_= False
    PCA_variability_ = 0.95
    with_ANOVA_ = False
    k_features_ = 3
    
    # Pipelines
    if with_PCA_:
        #Create PCA to redimention features to have X % variability
        pca = PCA(n_components=PCA_variability_, random_state=42)
    
        model = make_pipeline(pca, model)
        
    elif with_ANOVA_:
    
        #ANOVA filter, take best features
        anova_filter = SelectKBest(f_classif, k=k_features_)
    
        # Make Pipeline
        model = make_pipeline(anova_filter, model)
    
    return model

#%% Random Forest model for Gender

def genderRandomForest():
    # RandomForest Parameters 
    n_estimators_ = 250
    max_features_ = 'auto'
    max_depth_ = 10
    min_samples_leaf_ = 20
    bootstrap_=True
    with_ANOVA_ = False
    k_features_ = 3
    with_PCA_= False
    PCA_variability_ = 0.95
    
    randForest = RandomForestClassifier(n_estimators=n_estimators_,
                                        max_features=max_features_,
                                        max_depth= max_depth_,
                                        min_samples_leaf=min_samples_leaf_,
                                        bootstrap = bootstrap_,
                                        random_state=42)
        
    if with_PCA_:
            #Create PCA to redimention features to have X % variability
            pca = PCA(n_components=PCA_variability_, random_state=42)
        
            model = make_pipeline(pca, randForest)
        
    elif with_ANOVA_:
    
        #ANOVA filter, take 3 best features
        anova_filter = SelectKBest(f_classif, k=k_features_)
    
        # Make Pipeline
        model = make_pipeline(anova_filter, randForest)
    
    else:
        model = randForest
        
    return model

#%% MLP Classifier for Gender

def genderMLPClassifier():
    # Parameters for MLP classifier
    hidden_layer_sizes_=(100)
    activation_='relu'
    solver_='sgd'
    alpha_=.01
    learning_rate_='adaptive'
    early_stopping_ = True
    max_iter_=500
    with_ANOVA_ = False
    k_features_ = 3
    with_PCA_= False
    PCA_variability_ = 0.95
    
    mlpClass = MLPClassifier(hidden_layer_sizes= hidden_layer_sizes_, 
                             activation = activation_,  
                             solver=solver_, 
                             alpha = alpha_, 
                             learning_rate = learning_rate_ , 
                             max_iter=max_iter_, 
                             early_stopping = early_stopping_,
                             random_state=42)


    if with_PCA_:
        #Create PCA to redimention features to have X % variability
        pca = PCA(n_components=PCA_variability_, random_state=42)
    
        model = make_pipeline(pca, mlpClass)
    
    elif with_ANOVA_:
    
        #ANOVA filter, take 3 best features
        anova_filter = SelectKBest(f_classif, k=k_features_)
    
        # Make Pipeline
        model = make_pipeline(anova_filter, mlpClass)
    
    else:
        model = mlpClass
    
    return model

#%% Cross Validation for Train
start = timer()

metrics_tr = []
metrics_val = []

# cross validation for train
for number in range(10):

    language = ''
    mode = 'SubjectDependent'
    x_tr, y_tr, x_val, y_val,  = getDataset(number, language, mode)
    x_TS, y_TS = getTestDataset(language, mode)
    
    lanEnglish_tr = y_tr[:,4]
    lanEnglish_val = y_val[:,4]
    lanEnglish_TS = y_TS[:,4] #0 for English; 1 for Native

    y_tr = y_tr[:,1]
    y_val = y_val[:,1]
    y_TS = y_TS[:,1] #age


    #model = genderSVM()
    model = genderLogReg()
    model = genderSVM()
    #model = genderLogReg()

    #model = genderRandomForest()
    #model = genderMLPClassifier()
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

y_pred = model.predict(x_TS)
c=confusion_matrix(y_TS, y_pred)
df_cm = pd.DataFrame(c, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
 
plt.show()
end = timer()
print(f'{end-start:.3f}', 's')
