# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:20:14 2020

@author: m4ria

explaining short term features:

"The most important concept of audio feature extraction is short-term windowing (or framing):
this simply means that the audio signal is split into short-term windows (or frames).
The frames can be optionally overlapping.
The length of the frames usually ranges from 10 to 100msecs depending on the application and
types of signals. For the non-overlapping case, the step of the windowing procedure is equal
to the window's length (also called "size").
"""

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
import matplotlib.pyplot as plt
import pandas as pd

[Fs, x_stereo] = audioBasicIO.read_audio_file("S_01_4002_VV4.wav")
#convert the multichannel signal to single channel: needed for following functions to work!
x=audioBasicIO.stereo_to_mono(x_stereo)

#extraction of short term features
F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)

#joining features names with result
F=pd.DataFrame(F)
f_names=pd.DataFrame(f_names)
features=pd.concat([f_names,F], axis=1)

#rows will be the extracted features
#cols will be short-term frames that fit into the recording
pd.DataFrame(features).to_csv("C:/Users/m4ria/OneDrive/Documentos/4ano/DACO/ProjetoDACO/Dataset/test.csv")


#extras:
    
#MidTermFeatures.mid_feature_extraction_to_file('C:/Users/m4ria/OneDrive/Documentos/4ano/DACO/ProjetoDACO/Dataset/S_01_4002_VV4.wav', 1, 1, 0.5, 0.5, 'testfile' )

#F_mid, F_short, F_mid_names = MidTermFeatures.mid_feature_extraction(x, Fs, Fs, Fs, 0.05*Fs, 0.05*Fs)
