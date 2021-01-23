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

def identityClassification(open_set,allData,pool_Ids,subject_audio, thresh=''):
    
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
        
    
    probs= np.zeros((pool_Ids.shape[0],2))
    
    # Get scores for each model regarding subject in analysis
    for i in range(pool_Ids.shape[0]):
        model=models[i]
        prob=model.predict_proba(subject_audio[:,4:])[0]
        if prob.shape[0]==1:
            return 0
        probs[i,0]=prob[0]
        probs[i,1]=prob[1]
    
    if open_set:
        for i in range(pool_Ids.shape[0]):
            if probs[i,1]>thresh:
                return pool_Ids[i]
        return 0
    
    else:
        max_index = np.where(probs[:,1] == np.amax(probs[:,1]))
        if(len(max_index[0])!=1): #if two IDs have maximum probability, it is considered wrong
            return 0
        return pool_Ids[max_index]
    

    
#%% Get Dataset

allData = getAllDataset("Test")

# Define type of classification
open_set=False #if false, closed set will be performed
if open_set:
    print("OPEN-SET CLASSIFICATION \n")
else:
    print("CLOSED-SET CLASSIFICATION \n")
    
# Define number of tests
n_tests = 100
print('The classification will be tested ', n_tests, ' times\n')

# Define number of subjects that will integrate the pool
numSubjects = 10
print('The number of subjects in the pool is ', numSubjects, '\n')

# Define if text is audio dependent or not
audio_independent = False
print('Audio independency: ', audio_independent, '\n')

if open_set:
    # Define probability threshold
    thresh=0.7
    print('The subject will be considered identified if its probability is above ', thresh, '. \n')

scores=0
for i in range(n_tests):
    
    # Define subject and get its data
    
    # Choose random identity from the dataset
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
    
    # Randomly define subjects that will integrate the pool
    all_Ids = np.unique(allData[:,0])
    # Remove ID from dataset (will after be added)
    id_index = np.where(all_Ids == subject_Id)
    all_Ids = np.delete(all_Ids, (id_index), axis=0)
    np.random.shuffle(all_Ids)
    pool_Ids = all_Ids[0:numSubjects-1]
    
    # Add ID of subject in analysis
    pool_Ids=np.append(pool_Ids,subject_Id)
    np.random.shuffle(pool_Ids)

    # Identify Subject
    if open_set:
        identified_Id = identityClassification(open_set,allData,pool_Ids,subject_audio, thresh)
    else:
        identified_Id = identityClassification(open_set,allData,pool_Ids,subject_audio)
    
    # Evaluation
    if int(float(identified_Id)) == int(float(subject_Id)):
        scores=scores+1    

#%% Display Results

accuracy=scores/n_tests

print('The accuracy was', accuracy)
