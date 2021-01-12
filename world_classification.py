# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:10:08 2021

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
"""

import numpy as np
from data_load import getAllDataset
from sklearn.naive_bayes import GaussianNB

#%% Get Dataset
allData = getAllDataset("train")

#%% Define subject and get its data

#ALTERNATIVE 1: Choose random identity from the dataset
subject_Id = np.random.choice(allData[:,0],1)
# ALTERNATIVE 2: Define specific identity
#subject_Id = '4002.0'

# Get subjects data
audio_indexes=np.where(allData[:,0] == subject_Id)
subject_pool=allData[np.random.choice(audio_indexes[0],1),:]
subject_pool = np.array(subject_pool).astype(np.float)

#%% Define pool of subjects in analysis

numSubjects = 5

# ALTERNATIVE 1: Randomly define subjects that will integrate the pool
#all_Ids = np.unique(allData[:,0])
#pool_Ids = np.random.choice(all_Ids,numSubjects)
# ALTERNATIVE 2: Define specific number of subjects
pool_Ids=np.array(['4124.0','4160.0','4128.0','4001.0','4004.0'])
print("Random subject will be identified in a random pool with ", numSubjects, " subjects: \n")

# Get pool of audios
audio_pool = []
for id_index, id_value in enumerate(pool_Ids):
    for audio_index in range(allData.shape[0]):
        if allData[audio_index][0] == id_value:
            audio_pool.append(allData[audio_index][:])

audio_pool = np.array(audio_pool).astype(np.float)

#%% Train Gaussian NB Model for each identity

models=[]

for id_index, id_value in enumerate(pool_Ids):
    
    y=[]
    for i in range(audio_pool.shape[0]):
        if audio_pool[i,0]==int(float(id_value)):
            y.append(1)
        else:
            y.append(0)
    
    y = np.array(y).astype(np.float)
    #Train model of specific ID
    gauss = GaussianNB()
    models.append(gauss.fit(audio_pool[:,4:], y))

#%% Avaliate & Display Classification

# Get scores for each model regarding subject in analysis
scores=[]
for id in range(len(models)):
    model=models[id]
    scores.append(model.predict_proba(subject_pool[:,4:]))

# Verify subject's presence
isPresent = False
for id in range(len(scores)):
    
    if scores[id][0,1]==1:
        identified_subject=pool_Ids[id]
        isPresent=True
        break

# Print result
if isPresent:
    print("Subject in analysis was found in the pool and its ID number is ", identified_subject)
else:
    print("Subject was not found in pool.")
