# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:28:27 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
import numpy as np
from data_load import getDataset
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

#%% Load Datasets

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 5  
numValues = 2   # Mean, Standard Deviation

sMetrics_tr = np.zeros((numLanguages, numMetrics, numValues))
sMetrics_val = np.zeros((numLanguages, numMetrics, numValues))


# Define Parameters for Train
hidden_layer_sizes_=(50, 50, 50) #tentar 100 no meio
activation_='relu'
solver_='sgd'
alpha_=0.0001
learning_rate_='adaptive' #tentar constant
max_iter_=300
with_ANOVA_ = False
k_features_ = 23
with_PCA_= True
PCA_variability_ = 0.95


print('\nMLP Model \n')
print('PCA selection: ', with_PCA_)
print('max_iter =', max_iter_)
print('hidden_layer_sizes =', hidden_layer_sizes_)
print('activation =', activation_)
print('solver =', solver_)
print('alpha =', alpha_)
print('learning_rate =', learning_rate_)
print('PCA:', with_PCA_)
if with_PCA_:
    print('PCA variability:', PCA_variability_)
print('ANOVA:', with_ANOVA_)
if with_ANOVA_:
    print('k features:', k_features_)
  
for k, language in enumerate(languages):
    
    metrics_tr = []
    metrics_val = []
    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language)
        
        # Train SVM
        mlpClass = MLPClassifier(hidden_layer_sizes= hidden_layer_sizes_, activation = activation_,  solver=solver_, alpha = alpha_, learning_rate = learning_rate_ , max_iter=max_iter_, random_state=42)


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
            
        model.fit(x_tr, y_tr)
        """
        #Find best features to use in model:
        mlp = MLPClassifier(max_iter=300)

        parameter_space = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive', 'invscaling'],
        }
        
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        clf.fit(x_tr, y_tr)
        
        print(clf.best_params_)
        """
        
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

