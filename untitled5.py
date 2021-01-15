# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 01:51:52 2021

@author: User
"""
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from data_load import getTestDataset, getDataset, getAllDataset
from pickle_handling import getPickleFile
import numpy as np
from age_classification_3classes import ageProcessing

# aa = getPickleFile('datasetsNativeSubjectDependent')
# c,b =getTestDataset('Native', 'SubjectDependent')
# g,d,e,f =getDataset(3,'Native', 'SubjectDependent')
mode='SubjectIndependent'

x, y = getAllDataset('train')       


smote= SMOTE(random_state=2)

yProcessed = ageProcessing(y)

            
x_training, age_training = smote.fit_resample(x, yProcessed)  
plt.hist(yProcessed)
plt.hist(age_training)