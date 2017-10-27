from itertools import product
import numpy as np 
import pandas as pd 
import random

language=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_repository/language.csv")
accent=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_repository/accent.csv")

trsplit=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_splits/accent_train.txt",header=None)
vasplit=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_splits/accent_val.txt",header=None)

trsplit=trsplit.values.ravel()
vasplit=vasplit.values.ravel()

tr_acc=accent.loc[accent['Id'].isin(trsplit)]
va_acc=accent.loc[accent['Id'].isin(vasplit)]

tr_lan=language.loc[language['Id'].isin(trsplit)]

levels=list(set(tr_acc['Label']))
levels.sort()

accindex=[]
lanindex=[]
for level in levels:
	accindex+=[list(tr_acc.loc[tr_acc['Label']==level].index)]
	lanindex+=[list(tr_lan.loc[tr_lan['Label']==level].index)]


nsamples=5000
positive=[]
negative=[]
for i in range(len(levels)):
	for j in range(len(levels)):
		if(i==j):
			random.seed(100)
			positive+=random.sample(list(product(accindex[i],lanindex[j])),nsamples)
		else:
			random.seed(100)
			negative+=random.sample(list(product(accindex[i],lanindex[j])),nsamples)

len(positive)
len(negative)

pdf=pd.DataFrame(positive, columns=['acc_ind', 'lan_ind'])
ndf=pd.DataFrame(negative, columns=['acc_ind', 'lan_ind'])

pdf.to_csv('/home/aditya/Desktop/accent_identification2/siamese_network/split/val_train_pos.csv',index=False)
ndf.to_csv('/home/aditya/Desktop/accent_identification2/siamese_network/split/val_train_neg.csv',index=False)
