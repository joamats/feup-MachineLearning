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
def getMetrics(model_type, model, x, y):
    
    metrics=[]
    
    preds=model.predict(x) #get models results (0,1)
    
    metrics.append(model.score(x,y)) #get models accuracy
    
    metrics.append(f1_score(y,preds)) #get f1 score
    
    metrics.append(confusion_matrix(y, preds)) #get confusion matrix
    
    if (model_type=='withProbs'):
        probs = model.predict_proba(x)[:,1] #get models probabilities

        metrics.append(roc_auc_score(y,probs)) #get roc auc 

        metrics.append(average_precision_score(y, probs)) #get precision recall
    
    return metrics

#Displays all metrics
def displayMetrics(metrics):
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=metrics[2]).plot()
    
    print('The Accuracy of the model was ', f'{metrics[0]:.3f}')
    print('The F1 Score of the model was ' , f'{metrics[1]:.3f}')
 
    if(len(metrics)==5):
    
        print('The Area under the ROC curve of the model was ' , f'{metrics[3]:.3f}')
        
        print('The Precision Recall of the model was ', f'{metrics[4]:.3f}')
 