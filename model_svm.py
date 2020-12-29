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
X_tr, y_tr, X_val, y_val, X_ts, y_ts = getSubset(number, language)

#%% Train Models

model = svm.SVC(kernel = "poly")
model.fit(X_tr, y_tr)

#%% Assess Performance

score = model.score(X_val, y_val)
print(score)

#%%

# #%% Get and Display Model Metrics

# score, roc_auc, pr_auc, f1, conf_matrix = getMetrics(model, X_val, y_val)

# displayMetrics(score, roc_auc, pr_auc, f1, conf_matrix);