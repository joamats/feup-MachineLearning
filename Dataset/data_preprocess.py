# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 01:41:21 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

#%% Definition of auxiliary functions, for data split (by key)

# Returns a list where 1st line = training, 2nd = validation, 3rd = test
def train_val_test_split(X, tr_size=0.7):
    tr, aux = train_test_split(X, train_size=tr_size, shuffle=True)

    # nested hold out
    
    val, ts = train_test_split(aux, test_size=0.5, shuffle=True)
    
    
    return [tr, val, ts] 

# Returns subsets with unique subjects in each train, validation, test
def train_val_test_split_by_key(X, key, tr_size=0.6):
    subsets = train_val_test_split(X[key].unique())
    return [X.loc[X[key].isin(subset)] for subset in subsets]

#%% Definition of function that returns data split and pre-process
# to be called at the beginning of each model assessment

def getData():
    
    rawData = np.genfromtxt('dataset/extracted_features_all.csv', delimiter=',', encoding='utf8', dtype=np.str)
    rawSubjectsInfo = np.genfromtxt('dataset/subjects_information_all.csv', delimiter=';', encoding='utf8', dtype=np.str)
    
    # remove headers
    data = rawData[2:, 2:] 
    subjectsInfo = rawSubjectsInfo[1:,:]
    
    #%% Preparation of Infos Matrices
    
    # get the id of subjects, from the path name
    names = np.char.split(rawData[2:,1], '_')
    nameN = np.zeros((data.shape[0]))
    
    for i, name in enumerate(names):
        nameN[i] = name[2]
        
    nameN = nameN.reshape((data.shape[0]),1)
    
    # transform genders to numbers
    genders = np.unique(subjectsInfo[:,1])
    genders = genders.reshape((genders.shape[0],1))
    numGenders = np.arange(genders.shape[0])
    numGenders = numGenders.reshape(numGenders.shape[0],1)
    genders = np.concatenate((genders, numGenders), axis=1)
    
    for s in subjectsInfo:
        for g in genders:
            if s[1] == g[0]:
                s[1] = g[1]
                break
                    
            
    # transform languages to numbers
    languages = np.unique(subjectsInfo[:,3])
    languages = languages.reshape((languages.shape[0],1))
    numLanguages = np.arange(languages.shape[0])
    numLanguages = numLanguages.reshape(numLanguages.shape[0],1)
    languages = np.concatenate((languages,numLanguages ),axis=1)
    
    for s in subjectsInfo:
        for l in languages:
            if s[3] == l[0]:
                s[3] = l[1]
                break
            
    # add subjects info to data array
    infos = np.zeros((data.shape[0],3))
    
    for i, personId in enumerate(nameN[:,0]):
        for s in subjectsInfo:
            if(personId.astype(float) == s[0].astype(float)):
                infos[i,:] = s[1:]
                break
    
    #%% Concatenation of info data, according to language
    
    # empty lists initialization
    languageversion = []
    dataNativeLanguage = []
    dataEnglish = []
    
    
    # spearation of data language: english and native language
    for i, name in enumerate(names):
        languageversion.append(name[3]);
        if(name[3][1] == 'V'):
            dataNativeLanguage.append([nameN[i, 0], data[i,:], infos[i,:]])
        else:
            dataEnglish.append([nameN[i, 0], data[i,:], infos[i,:]])
       
    # conversion of lists to np.arrays 
    dataEnglish = np.array(dataEnglish, dtype = object)
    dataNativeLanguage = np.array(dataNativeLanguage, dtype = object)
    
    # conversion of np.arrays to pd.DataFrames
    dfdataEnglish = pd.DataFrame({'ID': dataEnglish[:,0],
                                  'Features': dataEnglish[:,1],
                                  'Infos': dataEnglish[:,2]})
    
    dfdataNativeLanguage = pd.DataFrame({'ID': dataNativeLanguage[:,0],
                                         'Features': dataNativeLanguage[:,1],
                                         'Infos': dataEnglish[:,2]})
    
    #%% Split of dataset: training, validation, and test
    
    
    # 10 random datasets with nested hold out, for english and native languages
    # persons are not repeated in train, validation and test (unique by key)
    
    datasetsEnglish = []
    datasetsNative = []
    
    # repeat shuffle process of data split 10x
    for i in range(10):
        datasetsEnglish.append(train_val_test_split_by_key(dfdataEnglish, 'ID', tr_size=0.6))
        datasetsNative.append(train_val_test_split_by_key(dfdataNativeLanguage, 'ID', tr_size=0.6))
    
    return datasetsEnglish, datasetsNative

