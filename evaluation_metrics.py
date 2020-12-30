# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:43:16 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

#Receives models inputs and the trained model, returns its metrics
def getMetrics(model, x, y, model_type=None):
    
    metrics=[]
    
    preds=model.predict(x) #get models results (0,1)
    
    metrics.append(model.score(x,y)) #get models accuracy
    
    metrics.append(f1_score(y,preds)) #get f1 score
    
    metrics.append(confusion_matrix(y, preds)) #get confusion matrix

    metrics.append(precision_score(y, preds)) #get precision score
    
    metrics.append(recall_score(y, preds)) #get recall score
    
    if (model_type == 'withProbs'):
        probs = model.predict_proba(x)[:,1] #get models probabilities

        metrics.append(roc_auc_score(y,probs)) #get roc auc 

    return metrics

#Displays all metrics
def displayMetrics(metrics):
    
    ConfusionMatrixDisplay(confusion_matrix=metrics[2]).plot()
    
    print('The Accuracy of the model was ', f'{metrics[0]:.3f}')
    print('The F1 Score of the model was ' , f'{metrics[1]:.3f}')
    print('The Precision Score of the model was ', f'{metrics[3]:.3f}')
    print('The Recall Score of the model was ' , f'{metrics[4]:.3f}')
 
    if(len(metrics)==6):
    
        print('The Area under the ROC curve of the model was ' , f'{metrics[3]:.3f}')

def getGeneralMetrics(metrics, numMetrics):
    
    sMetrics = np.zeros((numMetrics, 2))

    # Mean and Std for each metric
    for metric in range(numMetrics):    
    
        sMetrics[metric, 0] = np.mean([r[metric] for r in metrics])
        sMetrics[metric, 1] = np.std([r[metric] for r in metrics])
    
    return sMetrics

def displayGeneralMetrics(sMetrics):
    
    print('Accuracy =', f'{sMetrics[0,0]:.3f}', '+/-',  f'{sMetrics[0,1]:.3f}')
    print('F1-Score =', f'{sMetrics[1,0]:.3f}', '+/-', f'{sMetrics[1,1]:.3f}')
    print('Precision Score =', f'{sMetrics[3,0]:.3f}', '+/-', f'{sMetrics[3,1]:.3f}')
    print('Recall Score =', f'{sMetrics[4,0]:.3f}', '+/-', f'{sMetrics[4,1]:.3f}')
    
    if(len(sMetrics)==6):
        print('AUROC =', f'{sMetrics[5,0]:.3f}', '+/-', f'{sMetrics[5,1]:.3f}')
    
    