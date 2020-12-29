# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:45:59 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
    
Helping:
    
Access Set of 3 dataframes: datasetsEnglish[0]

Access Train Dataset: datasetsEnglish[0][0]

Access specfic column and row: datasetsEnglish[0][0]['Infos'][0]

Access element in row: datasetsEnglish[0][0]['Infos'][0]

"""

from evaluation_metrics import getMetrics, displayMetrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from data_load import getSubset

#%% Load Datasets

#include a for loop with zip here
number = 0
language = "English"
X_tr, y_tr, X_val, y_val, X_ts, y_ts = getSubset(number, language)

#%% Train Models

model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)

#%% Assess Performance

#Get Model Metrics
metrics = getMetrics('withProbs',model, X_val, y_val)

#Display Model Metrics
displayMetrics(metrics)


