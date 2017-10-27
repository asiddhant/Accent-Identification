import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle

#Load Model
model=pickle.load(open("/home/aditya/Desktop/accent_identification/accent_baselines/models/svm.p", "rb"))

accent=pd.read_csv("/home/aditya/Desktop/accent_identification/data_repository/accent.csv")

tesplit=pd.read_csv("/home/aditya/Desktop/accent_identification/data_splits/accent_test.txt",header=None)
tesplit=tesplit.values.ravel()
te_acc=accent.loc[accent['Id'].isin(tesplit)]

data=te_acc.values
data_x=data[:,1:401]
data_y=data[:,401]

outputclass=model.predict(data_x)
outputprobs=model.predict_proba(data_x)

# Test Representation
levels=list(set(accent['Label']))
levels.sort()

#Accuracy
print float(np.sum(outputclass==data_y))/(data_y.shape[0])
outputprobs=pd.DataFrame(outputprobs)
outputprobs.index=te_acc.index+1
outputprobs.columns=levels
outputprobs.to_csv('/home/aditya/Desktop/accent_identification/accent_baselines/files/svm.csv')

