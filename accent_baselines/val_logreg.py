import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import pickle

# Loading the Data and Splitting into Train-Val
accent=pd.read_csv("/home/aditya/Desktop/accent_identification/data_repository/accent.csv")

trsplit=pd.read_csv("/home/aditya/Desktop/accent_identification/data_splits/accent_train.txt",header=None)
vasplit=pd.read_csv("/home/aditya/Desktop/accent_identification/data_splits/accent_val.txt",header=None)
trsplit=trsplit.values.ravel()
vasplit=vasplit.values.ravel()

tr_acc=accent.loc[accent['Id'].isin(trsplit)].values
va_acc=accent.loc[accent['Id'].isin(vasplit)].values

data=np.concatenate((tr_acc,va_acc),0)

data_x=data[:,1:401]
data_y=data[:,401]

# SVM Classifier
clf = linear_model.LogisticRegression()
pdcv = PredefinedSplit(test_fold=[-1]*len(trsplit)+[0]*len(vasplit))
param_grid = dict(penalty=['l1','l2'],C=[1,2,4,8])

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=pdcv, n_jobs=6)
grid_result = grid.fit(data_x,data_y)

f1=open('/home/aditya/Desktop/accent_identification/accent_baselines/logfiles/logreg.txt', 'a+')

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
	f1.write("%f with: %r \n" % (mean, param))

f1.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
f1.close()

pickle.dump(grid,open("/home/aditya/Desktop/accent_identification/accent_baselines/models/logreg.p","wb"))

