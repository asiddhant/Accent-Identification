import numpy as np
import pandas as pd 
from sklearn import discriminant_analysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import pickle

# Loading the Data and Splitting into Train-Val
language=pd.read_csv("/home/aditya/Desktop/accent_identification/data_repository/language.csv")

trsplit=pd.read_csv("/home/aditya/Desktop/accent_identification/data_splits/language_train.txt",header=None)
vasplit=pd.read_csv("/home/aditya/Desktop/accent_identification/data_splits/language_val.txt",header=None)
trsplit=trsplit.values.ravel()
vasplit=vasplit.values.ravel()

tr_lan=language.loc[language['Id'].isin(trsplit)].values
va_lan=language.loc[language['Id'].isin(vasplit)].values

data=np.concatenate((tr_lan,va_lan),0)

data_x=data[:,1:401]
data_y=data[:,401]

# SVM Classifier
clf = discriminant_analysis.LinearDiscriminantAnalysis()
pdcv = PredefinedSplit(test_fold=[-1]*len(tr_lan)+[0]*len(va_lan))
param_grid = dict(n_components=[7,8,9])

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=pdcv, n_jobs=6)
grid_result = grid.fit(data_x,data_y)

f1=open('/home/aditya/Desktop/accent_identification/language_baselines/logfiles/lda.txt', 'a+')

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
	f1.write("%f with: %r \n" % (mean, param))

f1.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
f1.close()

pickle.dump(grid,open("/home/aditya/Desktop/accent_identification/language_baselines/models/lda.p","wb"))

