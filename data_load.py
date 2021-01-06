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
def getDataset(number, language):
    
    filename="datasets" + language
    
    allDatasets = getPickleFile(filename)

    dataset = allDatasets[number]
    
    # train data
    x_tr = dataset[0][:,4:]
    y_tr = dataset[0][:,1]

    # validation data
    x_val = dataset[1][:,4:]
    y_val = dataset[1][:,1]

    # test data
    x_ts = dataset[2][:,4:]
    y_ts = dataset[2][:,1]
        
    return x_tr, y_tr, x_val, y_val, x_ts, y_ts

#%% Definition of function to join datasets of both languages

def getAllDataset(dataset_portion):
    
    filename="allData"

    allData = getPickleFile(filename)
    
    if dataset_portion == "train":
        # train data
        allData = allData[0]
    else: 
        # test data
        allData = allData[1]
     
    return allData
