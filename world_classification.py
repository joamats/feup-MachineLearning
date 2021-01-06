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

languages = ['Native', 'English']
numLanguages = len(languages)
        
allData = getAllDataset("train")

# Define number of subject pool and subject to identify
numSubjects = 10
subject_Id = 4125
print('Subject number ', subject_Id, "will be identified in a random pool with ", numSubjects, " subjects.")

# Randomly define the subjects that will integrate the pool
all_Ids = np.unique(allData[:,0])
pool_Ids = np.random.choice(all_Ids,numSubjects)

audio_pool = list(range(pool_Ids.shape[0]))

# Restrict dataset by choosing N random identities
for id_index, id_value in enumerate(pool_Ids):
    for audio_index in range(allData.shape[0]):
        if allData[audio_index][0] == id_value:
            audio_pool[id_index].append(allData[audio_index][:])

audio_pool=np.array(audio_pool)

# Calculate Gaussian Multivariate Parameters for each subject
#for audio_index, audio_id in enumerate(audio_pool):
   
    #covariance matrix
    #np.cov(data, rowvar=True)
    
    #calculate mean
