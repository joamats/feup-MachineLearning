# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 01:41:21 2020

@author: User
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


rawData = np.genfromtxt('extracted_features_all.csv', delimiter=',', encoding='utf8', dtype=np.str)
rawSubjectsInfo = np.genfromtxt('subjects_information_all.csv', delimiter=';', encoding='utf8', dtype=np.str)

dataNativeLanguage = []
dataEnglish = []

# remove headers
data = rawData[2:, 2:] 
subjectsInfo = rawSubjectsInfo[1:,:]
names = np.char.split(rawData[2:,1], '_')
nameN = np.zeros((data.shape[0]))

for i, name in enumerate(names):
    nameN[i] = name[2]
    
nameN = nameN.reshape((data.shape[0]),1) #ids of every audio

# get the id of subjects, from the path name

languageversion = []

#separating the data from english and native language
for i, name in enumerate(names):
    languageversion.append(name[3]);
    if(name[3][1] == 'V'):
        dataNativeLanguage.append([nameN[i, 0], data[i,:]])
    else:
        dataEnglish.append([nameN[i, 0], data[i,:]])
   
dataEnglish = np.array(dataEnglish, dtype = object)
dfdataEnglish = pd.DataFrame({'ID': dataEnglish[:,0], 'Features': dataEnglish[:,1]})
dataNativeLanguage = np.array(dataNativeLanguage, dtype = object)
dfdataNativeLanguage = pd.DataFrame({'ID': dataNativeLanguage[:,0], 'Features': dataNativeLanguage[:,1]})

#separating the datasets by training, validation and test

def train_val_test_split(X, tr_size=0.7):
    tr, aux = train_test_split(X, train_size=tr_size, shuffle=True)

    #nested hold out
    
    val, ts = train_test_split(aux, test_size=0.5, shuffle=True)
    
    return [tr, val, ts] #returns a list where first line= training, second = validation, third = teste

def train_val_test_split_by_key(X, key, tr_size=0.6):
    subsets = train_val_test_split(X[key].unique())
    return [X.loc[X[key].isin(subset)] for subset in subsets]

#final datasets in english languages
datasetE1 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE2 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE3 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE4 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE5 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE6 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE7 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE8 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE9 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)
datasetE10 = train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6)

#final datasets in native languages
datasetN1 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN2 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN3 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN4 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN5 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN6 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN7 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN8 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN9 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)
datasetN10 = train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6)