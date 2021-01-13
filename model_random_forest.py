# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:45:59 2020

@authors:
    João Matos
    Maria Loureiro
    Maria Carvalho
    
    
n_estimators: in general the more trees the less likely the algorithm is to overfit. So try increasing this. The lower this number, the closer the model is to a decision tree, with a restricted feature set.
max_features: try reducing this number (try 30-50% of the number of features). This determines how many features each tree is randomly assigned. The smaller, the less likely to overfit, but too small will start to introduce under fitting.
max_depth: Experiment with this. This will reduce the complexity of the learned models, lowering over fitting risk. Try starting small, say 5-10, and increasing you get the best result.
min_samples_leaf: Try setting this to values greater than one. This has a similar effect to the max_depth parameter, it means the branch will stop splitting once the leaves have that number of samples each

"""

from evaluation_metrics import getMetrics, getGeneralMetrics, displayGeneralMetrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from data_load import getDataset
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

#%% Load Datasets

languages = ['Native', 'English']
numLanguages = len(languages)
numMetrics = 5 # Accuracy, F1-score, precision score, recall score, roc_auc  
numValues = 2   # Mean, Standard Deviation

sMetrics_tr = np.zeros((numLanguages, numMetrics, numValues))
sMetrics_val = np.zeros((numLanguages, numMetrics, numValues))


# Define Parameters for Train
n_estimators_ = 200
max_features_ = 'auto'
max_depth_ = 10
min_samples_leaf_ = 20
bootstrap_=True
with_ANOVA_ = False
k_features_ = 23
with_PCA_= True
PCA_variability_ = 0.95

print('\nRandom Forest Model \n')
print('n_estimators =', n_estimators_)
print('max_features =', max_features_)
print('max_depth =', max_depth_)
print('min_samples_leaf =', min_samples_leaf_)
print('bootstrap =', bootstrap_)
print('PCA:', with_PCA_)
if with_PCA_:
    print('PCA variability:', PCA_variability_)
print('ANOVA:', with_ANOVA_)
if with_ANOVA_:
    print('k features:', k_features_)
  
    
for k, language in enumerate(languages):
    
    metrics_tr = []
    metrics_val = []
    
    for number in range(10):
        
        # Get *this* dataset
        x_tr, y_tr, x_val, y_val = getDataset(number, language)
        
        #getting only the gender labels
        y_tr = y_tr[:,0] 
        y_val = y_val[:,0]
        # Train SVM
        randForest = RandomForestClassifier(n_estimators=n_estimators_, max_features=max_features_, max_depth= max_depth_, min_samples_leaf=min_samples_leaf_, bootstrap = bootstrap_, random_state=42)
        
        if with_PCA_:
            #Create PCA to redimention features to have X % variability
            pca = PCA(n_components=PCA_variability_, random_state=42)
        
            model = make_pipeline(pca, randForest)
        
        elif with_ANOVA_:
        
            #ANOVA filter, take 3 best features
            anova_filter = SelectKBest(f_classif, k=k_features_)
        
            # Make Pipeline
            model = make_pipeline(anova_filter, randForest)
        
        else:
            model = randForest
            
        model.fit(x_tr, y_tr)
        
        """
        #Find best features to use in model:
        rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

        param_grid = { 
            'n_estimators': [50, 100, 200, 700, 1000],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [5, 10, 20, 30],
            'min_samples_leaf': [20, 50, 100]
        }

        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        CV_rfc.fit(x_tr, y_tr)
        
        print (CV_rfc.best_params_)
        
        """
        
        # Assess *this* model
        metrics_tr.append(getMetrics(model, x_tr, y_tr, 'withProbs'))
        metrics_val.append(getMetrics(model, x_val, y_val, 'withProbs'))

    #get mean and std for each metric
    sMetrics_tr = getGeneralMetrics(metrics_tr, numMetrics)
    sMetrics_val = getGeneralMetrics(metrics_val, numMetrics)

    print('\nLanguage:', language,'\n')
    print('Training Set')
    displayGeneralMetrics(sMetrics_tr)
    print('\n')
    print('Validation Set')
    displayGeneralMetrics(sMetrics_val)
    print('\n')

