# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from sklearn import svm
from data_load import getDataset
from evaluation_metrics import getMetrics, displayMetrics

#%% Load Datasets

#include a for loop with zip here
number = 9
language = "Native"
x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language)

#%% Train Models

model = svm.SVC(kernel = "poly")
model.fit(x_tr, y_tr)

#%% Assess Performance

metrics = getMetrics(model, x_val, y_val)

displayMetrics(metrics)