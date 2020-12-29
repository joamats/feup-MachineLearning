# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 19:26:05 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

import pickle

#%% Definition of function that creates pickle file

def createPickleFile(variable, pickleName):
    PIK = pickleName + ".dat"

    with open(PIK, "wb") as f:
        pickle.dump(variable, f)
        

#%% Definition of function that gets pickle file

def getPickleFile(pickleName):
    PIK = pickleName + ".dat"

    with open(PIK, "rb") as f:
        return(pickle.load(f))
        
