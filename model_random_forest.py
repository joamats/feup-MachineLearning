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

from model_metrics import getMetrics, displayMetrics
from sklearn.ensemble import RandomForestClassifier
from pickle_handling import getPickleFile
import numpy as np
import pandas as pd

#%% Get all necessary variables to train the model

datasetsEnglish = getPickleFile("datasetsEnglish")
#depois adicionar aqui um ciclo for para treinar tds os modelos
#por agora testo só com o primeiro set:
#datasetsEnglish = datasetsEnglish[0] 

"""
x_tr=datasetsEnglish[0]['Features']
y_tr=datasetsEnglish[0]['Gender'][0]
    
x_val=datasetsEnglish[1]['Features']
y_val=datasetsEnglish[1]['Gender'][0]

x_ts=datasetsEnglish[2]['Features']
y_ts=datasetsEnglish[2]['Gender'][0]

#Train Model
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(x_tr, y_tr)

#Get Model Metrics
score, roc_auc, pr_auc, f1, conf_matrix = getMetrics(model, x_val, y_val)

#Display Model Metrics
displayMetrics(score, roc_auc, pr_auc, f1, conf_matrix)

"""