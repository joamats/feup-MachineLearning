# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from sklearn import svm
from data_load import getSubset
from evaluation_metrics import getMetrics, displayMetrics

#%% Load Datasets

#include a for loop with zip here
number = 0
language = "English"
x_tr, y_tr, x_val, y_val, x_ts, y_ts = getSubset(number, language)

#%% Train Models

<<<<<<< HEAD
model = svm.SVC(kernel = "poly")
model.fit(x_tr, y_tr)
=======
model = svm.SVC(kernel = "poly").fit(X_tr, y_tr)
>>>>>>> fa57cd469c3c28ba70e02616cf40d44543207b1b

#%% Assess Performance

score = model.score(x_val, y_val)
print(score)

#%%

# #%% Get and Display Model Metrics

# score, roc_auc, pr_auc, f1, conf_matrix = getMetrics(model, X_val, y_val)

# displayMetrics(score, roc_auc, pr_auc, f1, conf_matrix);