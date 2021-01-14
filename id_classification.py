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

def worldClassification(allData,pool_Ids,subject_audio, thresh):
    
    # Get pool of audios
    audio_pool = []
    for id_index, id_value in enumerate(pool_Ids):
        for audio_index in range(allData.shape[0]):
            if allData[audio_index][0] == id_value:
                audio_pool.append(allData[audio_index][:])
    
    audio_pool = np.array(audio_pool).astype(np.float)
    
    # Train Gaussian NB Model for each identity
    
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
        
    probs=[]
    # Get scores for each model regarding subject in analysis
    for i in range(pool_Ids.shape[0]):
        model=models[i]
        probs.append(model.predict_proba(subject_audio[:,4:]))

    for i in range(pool_Ids.shape[0]):
        if(probs[i][0].shape[0])==1: #REMOVE
            print(probs)
            print(pool_Ids[i])
        if probs[i][:,1]>thresh:
            return pool_Ids[i]
        
    return 0

    
#%% Get Dataset
allData = getAllDataset("Test")

# Define number of tests
n_tests = 100
print('The classification will be tested ', n_tests, ' times. \n')

# Define number of subjects that will integrate the pool
numSubjects = 100
print('The number of subjects in the pool is ', numSubjects, '. \n')

# Define if text is audio dependent or not
audio_independent = False
print('Audio independency: ', audio_independent, '. \n')

# Define probability threshold
thresh=0.7
print('The subject will be considered identified if its probability is above ', thresh, '. \n')

scores=0

for i in range(n_tests):
    
    # Define subject and get its data
    
    #ALTERNATIVE 1: Choose random identity from the dataset
    subject_Id = np.random.choice(allData[:,0],1)
    
    # Get subjects data
    audio_indexes = np.where(allData[:,0] == subject_Id)
    index=np.random.choice(audio_indexes[0],1)
    subject_audio = allData[index,:]
    subject_audio = np.array(subject_audio).astype(np.float)
    # Remove audio from dataset
    if audio_independent:
        allData = np.delete(allData, (index), axis=0)
    
    # Define pool of subjects in analysis
    
    # ALTERNATIVE 1: Randomly define subjects that will integrate the pool
    all_Ids = np.unique(allData[:,0])
    np.random.shuffle(all_Ids)
    pool_Ids = all_Ids[0:numSubjects-1]
    
    # Closed Set Classification
    pool_Ids=np.append(pool_Ids,subject_Id)
    np.random.shuffle(pool_Ids)

    # Identify Subject
    identified_Id = worldClassification(allData,pool_Ids,subject_audio, thresh)
    
    # Evaluation
    if int(float(identified_Id)) == int(float(subject_Id)):
        scores=scores+1
    

#%% Display Results

accuracy=scores/n_tests

print('The accuracy was', accuracy)
