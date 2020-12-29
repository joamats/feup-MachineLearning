# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:43:16 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, average_precision_score

#Receives models inputs and the trained model, returns its metrics
def getMetrics(model, x, y):
    
    probs = model.predict_proba(x)[:,1] #get models probabilities

    preds = model.predict(x) #get models results (0,1)

    score = model.score(x,y) #get models accuracy

    roc_auc = roc_auc_score(y,probs) #get roc auc 

    pr_auc = average_precision_score(y, probs) #get precision recall

    f1 = f1_score(y,preds) #get f1 score
    
    conf_matrix = confusion_matrix(y, preds) #get confusion matrix
    
    return score, roc_auc, pr_auc, f1, conf_matrix

#Displays all metrics
def displayMetrics(score, roc_auc, pr_auc, f1, conf_matrix):
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot()
    print('The Accuracy of the model was ', f'{score:.3f}')
    print('The Area under the ROC curve of the model was ' , f'{roc_auc:.3f}')
    print('The Precision Recall of the model was ', f'{pr_auc:.3f}')
    print('The F1 Score of the model was ' , f'{f1:.3f}')
 