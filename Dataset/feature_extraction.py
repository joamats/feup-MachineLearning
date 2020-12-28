# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:20:14 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho

"""
from pickle_handling import createPickleFile
from pyAudioAnalysis import MidTermFeatures
import pandas as pd
import numpy as np

# Extract the mid-term features of the WAVE files of a particular folder
#Windows and steps based on following example:
#https://medium.com/behavioral-signals-ai/intro-to-audio-analysis-recognizing-sounds-using-machine-learning-20fd646a0ec5
mid_term_features, wav_file_list2, mid_feature_names = MidTermFeatures.directory_feature_extraction('signals', 1, 1, 0.1, 0.05,compute_beat=False)


# Resize mid_features_names to add to dataframe
mid_feature_names = np.array(mid_feature_names)
mid_feature_names=mid_feature_names.reshape((mid_feature_names.shape[0],1))
mid_feature_names=np.transpose(mid_feature_names)

# Resize wav_file_list2 to add to dataframe
wav_file_list2 = np.array(wav_file_list2)
wav_file_list2=np.insert(wav_file_list2,0,'file_path')
wav_file_list2 = wav_file_list2.reshape((wav_file_list2.shape[0],1))

# Add all arrays and create dataframe
mid_term_features = np.vstack([mid_feature_names, mid_term_features])
mid_term_features=np.append(wav_file_list2, mid_term_features, axis=1)
df=pd.DataFrame(mid_term_features)

#save dataframe in pickle
createPickleFile(df, "extracted_features")

