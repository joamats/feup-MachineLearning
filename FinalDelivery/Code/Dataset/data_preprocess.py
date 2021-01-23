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
from pickle_handling import createPickleFile, getPickleFile
from collections import Counter

#%% Definition of auxiliary functions, for data split (by key)

# Returns a list where 1st line = training, 2nd = validation, 3rd = test
def train_val_test_split(X):
    
     aux, ts = train_test_split(X, test_size=0.15, shuffle=True)

     tr = []
     val =[]
    # nested hold out
     for i in range(10): # will return 10 sets of validation and train data
        train, valid = train_test_split(aux, test_size = 0.18, shuffle=True)
        tr.append(train)
        val.append(valid)
    
     return [tr, val], ts

# Returns subsets with unique subjects in each train, validation, test
def train_val_test_split_by_key(X, key):
    subsetTV, subsets = train_val_test_split(np.unique(X[:,key]))
    return [X[np.isin(X[:,key],subset)] for subset in subsetTV[0]], [X[np.isin(X[:,key],subset)] for subset in subsetTV[1]], X[np.isin(X[:,key],subsets)] 

# Returns subsets subject dependent in each train, validation, test
def train_val_test_split_not_by_key(X):
    subsetTV, subsets = train_val_test_split(X)
    return subsetTV[0], subsetTV[1], subsets


#%% Definition of function that returns data split and pre-process
# to be saved in pickle format

def getData():
    
    rawData = getPickleFile("extracted_features")
    rawSubjectsInfo = np.genfromtxt('subjects_information_all.csv', delimiter=';', encoding='utf8', dtype=np.str)
    
    # from dataframe to numpy
    rawData = pd.DataFrame(rawData).to_numpy()
    rawData = np.array(rawData, dtype = str)
    
    # remove headers
    subjectsInfo = rawSubjectsInfo[1:,:]
    data = rawData[1:, 1:] 

    #%% Preparation of Infos Matrices
    
    # get the id of subjects, from the path name
    names = np.char.split(rawData[1:,0], '_')
    nameN = np.zeros((data.shape[0]))
    nameL = []
    
    for i, name in enumerate(names):
        nameN[i] = name[2]    # person ID
        nameL.append(name[3][1]) # audio language

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
    
    data = np.concatenate((nameN, infos, data), axis=1)
    
    # get number of features
    numFeatures = data.shape[1]
    
    # get number of language audios
    c = Counter(nameL)
    numEnglish = c['E']
    numNative = c['V']
    
    dataNative = np.zeros((numNative, numFeatures))
    dataEnglish = np.zeros((numEnglish, numFeatures))
    
    i_e = 0
    i_n = 0

    for i, l in enumerate(nameL):
        if l == 'E':
            dataEnglish[i_e,:] = data[i,:]
            i_e += 1
            nameL[i] = 0 #transform nameL to 0 if english
        else:
            dataNative[i_n,:] = data[i,:]
            i_n += 1
            nameL[i] = 1 #transform nameL to 1 if native
        
    nameL = np.array(nameL)  
    nameL = nameL.reshape(nameL.shape[0],1)
    data = np.concatenate((data[:,:4], nameL, data[:,4:]), axis=1).astype(float)
   #%% Concatenation of info data, according to gender
    
    # get number of features
    numFeatures = data.shape[1]
    
    maleData = []
    femaleData = []
  
    for i in range(data.shape[0]):
        if data[i,1]==0:
            femaleData.append(data[i,:])
        else:
            maleData.append(data[i,:])
        
    maleData = np.array(maleData)  
    femaleData = np.array(femaleData)
    
    
    #%% Split of dataset: training, validation, and test
    
    # # 10 random datasets with nested hold out, for english and native languages
    # # persons are not repeated in train, validation and test (unique by key)
    
    allDatasetsSubjectIndependent = train_val_test_split_by_key(data, 0)
    allDatasetsSubjectDependent = train_val_test_split_not_by_key(data)
    
    maleDataSubjectIndependent = train_val_test_split_by_key(maleData, 0)
    maleDataSubjectDependent = train_val_test_split_not_by_key(maleData)
    femaleDataSubjectIndependent = train_val_test_split_by_key(femaleData, 0)
    femaleDataSubjectDependent = train_val_test_split_not_by_key(femaleData)
    
    return allDatasetsSubjectDependent, allDatasetsSubjectIndependent, maleDataSubjectDependent,maleDataSubjectIndependent,femaleDataSubjectDependent, femaleDataSubjectIndependent


#%% Create pickle files with all necessary variables

[allDatasetsSubjectDependent, allDatasetsSubjectIndependent, maleDataSubjectDependent,maleDataSubjectIndependent,femaleDataSubjectDependent, femaleDataSubjectIndependent] = getData()

#create datasets in root

createPickleFile(allDatasetsSubjectDependent, '../datasetsSubjectDependent')
createPickleFile(allDatasetsSubjectIndependent, '../datasetsSubjectIndependent')

createPickleFile(maleDataSubjectIndependent, '../datasetsMaleSubjectIndependent')
createPickleFile(maleDataSubjectDependent, '../datasetsMaleSubjectDependent')

createPickleFile(femaleDataSubjectIndependent, '../datasetsFealeSubjectIndependent')
createPickleFile(femaleDataSubjectDependent, '../datasetsFemaleSubjectDependent')
