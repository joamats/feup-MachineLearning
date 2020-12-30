# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 01:10:02 2020

@author: João Matos
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
from data_load import getDataset

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
languages = ['Native', 'English']   
for k, language in enumerate(languages):
    for number in range(10): 
        
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = getDataset(number, language, True)
        
 
        # feature selection
        x_tr_fs, x_val_fs, fs = select_features(x_tr, y_tr, x_val)
        # what are scores for the features
        for i in range(len(fs.scores_)):
        	print('Feature %d: %f' % (i, fs.scores_[i]))
        # plot the scores
        pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
        pyplot.show()