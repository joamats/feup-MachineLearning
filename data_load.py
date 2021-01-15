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
# returns X and y for tr, val
def getDataset(number, language, mode, gender=''):
    
    filename="datasets" + language + gender + mode
    
    allDatasets = getPickleFile(filename)

    trainDataset = allDatasets[0]
    validationDataset = allDatasets[1]
    
    # train data
    x_tr = trainDataset[number][:,5:]
    y_tr = trainDataset[number][:,:5]

    # validation data
    x_val = validationDataset[number][:,5:]
    y_val = validationDataset[number][:,:5]


        
    return x_tr, y_tr, x_val, y_val
#%% returns X and y for ts

def getTestDataset(language, mode, gender=''):
     filename="datasets" + language + gender + mode     
     allDatasets = getPickleFile(filename)
     
     # test data
     x_ts = allDatasets[2][:,5:]
     y_ts = allDatasets[2][:,:5]
     
     return x_ts, y_ts
#%% Definition of function to join datasets of both languages

def getAllDataset(dataset_portion):
    
    filename="allData"

    allData = getPickleFile(filename)
    
    if dataset_portion == "train":
        # train data
        allData = allData[0]
        x= allData[:,4:]
        y = allData[:,2]
    else: 
        # test data
        allData = allData[1]
        
    return x,y
