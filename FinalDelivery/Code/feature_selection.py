# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 01:10:02 2020

@author: João Matos
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot 

# feature selection
def select_features(x, y):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(x, y)
	
	return  fs

def plot_score_features(x,y):
 
    # feature selection
    fs = select_features(x,y)

    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.xlabel("Features")
    pyplot.ylabel("ANOVA Score")
    
    pyplot.show()    
