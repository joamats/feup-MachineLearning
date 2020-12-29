# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from pickle_handling import getPickleFile
import numpy as np

#%% Definition of function to load data and return subset with tr, val, ts

# receives number (from 0 to 9) and language("English" or "Native")
# returns X and y for tr, val, ts
def getSubset(number, language):
    
    allDatasets = getPickleFile("datasets" + language)

    datasetEnglish = allDatasets[number]
    
    # train data
    x_tr = datasetEnglish[0]['Features']
    y_tr = datasetEnglish[0]['Gender']
    
    # conversion to numpy correct format
    X_tr = np.zeros((x_tr.shape[0], x_tr.iloc[0].shape[0]))
    for i, d in enumerate(x_tr):
        X_tr[i,:] = d
    
    # validation data
    x_val = datasetEnglish[1]['Features']
    y_val = datasetEnglish[1]['Gender']
    
    # conversion to numpy correct format
    X_val = np.zeros((x_val.shape[0], x_val.iloc[0].shape[0]))
    for i, d in enumerate(x_val):
        X_val[i,:] = d
    
    # test data
    x_ts = datasetEnglish[2]['Features']
    y_ts = datasetEnglish[2]['Gender']
    
    # conversion to numpy correct format
    X_ts = np.zeros((x_ts.shape[0], x_ts.iloc[0].shape[0]))
    for i, d in enumerate(x_ts):
        X_ts[i,:] = d
    
    return X_tr, y_tr, X_val, y_val, X_ts, y_ts