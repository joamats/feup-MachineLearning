# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:19:17 2020

@author: João Matos
"""

import numpy as np

rawData = np.genfromtxt('extracted_features_all.csv', delimiter=',', encoding='utf8', dtype=np.str)
rawSubjectsInfo = np.genfromtxt('subjects_information_all.csv', delimiter=';', encoding='utf8', dtype=np.str)

# remove headers
data = rawData[2:, 2:] 
subjectsInfo = rawSubjectsInfo[1:,:]

# get the id of subjects, from the path name
names = np.char.split(rawData[2:,1], '_')
nameN = np.zeros((data.shape[0]))

for i, name in enumerate(names):
    nameN[i] = name[2]
    
nameN = nameN.reshape((data.shape[0]),1)
data = np.concatenate((data, nameN), axis=1) 


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
                
# to-do: create classes for ages!      
        
# transform languages to numbers
languages = np.unique(subjectsInfo[:,3])
languages = languages.reshape((languages.shape[0],1))
numLanguages = np.arange(languages.shape[0])
numLanguages = numLanguages.reshape(numLanguages.shape[0],1)
languages = np.concatenate((languages,numLanguages ),axis=1)

# transform languages to numbers
for s in subjectsInfo:
    for l in languages:
        if s[3] == l[0]:
            s[3] = l[1]
            break
        
# include subjects info in data array
infos = np.zeros((data.shape[0],3))

for i, personId in enumerate(data[:,-1]):
    for s in subjectsInfo:
        if(personId.astype(float) == s[0].astype(float)):
            infos[i,:] = s[1:]
            break

# concatenate infos to the end of data array
data = np.concatenate((data, infos), axis=1)

        
        
        

    
