# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:45:59 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

import pickle


#Get all necessary variable to train the model
PIK="datasetsEnglish.dat"
with open(PIK, "rb") as f:
    datasetsEnglish=pickle.load(f)
