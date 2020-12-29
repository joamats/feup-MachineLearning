# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:39:42 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from pickle_handling import getPickleFile

datasetsEnglish = getPickleFile("datasetsEnglish")
datasetsNative = getPickleFile("datasetsNative")

"""

#datasetsEnglish = datasetsEnglish[0] 

x_tr=datasetsEnglish[0]['Features']
y_tr=datasetsEnglish[0]['Gender'][0]

# x_val=datasetsEnglish[1]['Features']
# y_val=datasetsEnglish[1]['Gender'][0]

# x_ts=datasetsEnglish[2]['Features']
# y_ts=datasetsEnglish[2]['Gender'][0]

# #Train Model
# model = RandomForestClassifier(n_estimators=20, random_state=0).fit(x_tr, y_tr)

# #Get Model Metrics
# score, roc_auc, pr_auc, f1, conf_matrix = getMetrics(model, x_val, y_val)


#Display Model Metrics
displayMetrics(score, roc_auc, pr_auc, f1, conf_matrix)

"""