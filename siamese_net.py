'''Train a Siamese MLP on pairs of language and accent i-Vectors on 
the CSLU Accented Speech and 22 Lang dataset.'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

seed=1
np.random.seed(seed)

import random
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop,Adam
from keras import backend as K
from itertools import product


#Utility Functions
def euclidean_dist(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

def custom_accuracy(y_true, y_pred):
    return K.mean(K.equal((y_pred>0.5),y_true))


def create_shared_network(input_dim):
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='sigmoid'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='sigmoid'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='sigmoid'))
    return seq


def compute_accuracy(labels, predictions):
    return np.mean(labels==(predictions.ravel() > 0.5))

# Params
input_dim = 400
epochs = 30


# Data Loading
language=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_repository/language.csv")
accent=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_repository/accent.csv")

trsplit=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_splits/accent_train.txt",header=None)
vasplit=pd.read_csv("/home/aditya/Desktop/accent_identification2/data_splits/accent_val.txt",header=None)
trsplit=trsplit.values.ravel()
vasplit=vasplit.values.ravel()
tr_acc=accent.loc[accent['Id'].isin(trsplit)]
va_acc=accent.loc[accent['Id'].isin(vasplit)]

tr_lan=language.loc[language['Id'].isin(trsplit)]

# Test Representation
levels=list(set(tr_acc['Label']))
levels.sort()

rep_lan=[]

for level in levels:
    rep_lan+=[[level]+list(np.mean(tr_lan.loc[tr_lan['Label']==level].values[:,1:401],0))+[level]]

rep_lan=pd.DataFrame(rep_lan)
rep_lan.columns=['Id']+['V'+str(i) for i in range(1,401)]+['Label']

pos = pd.read_csv("/home/aditya/Desktop/accent_identification2/siamese_network/split/val_train_pos.csv")
neg = pd.read_csv("/home/aditya/Desktop/accent_identification2/siamese_network/split/val_train_neg.csv")

pos = pos.values
neg = neg.values

pos_a = accent.values[pos[:,0],1:401]
pos_b = language.values[pos[:,1],1:401]
pos_y = np.zeros(pos.shape[0])

neg_a = accent.values[neg[:,0],1:401]
neg_b = language.values[neg[:,1],1:401]
neg_y = np.ones(neg.shape[0])

tr_a = np.concatenate((pos_a,neg_a),axis=0)
tr_b = np.concatenate((pos_b,neg_b),axis=0)
tr_y = np.concatenate((pos_y,neg_y),axis=0)

random.seed(100)
index = range(tr_y.shape[0])
random.shuffle(index)

tr_a = tr_a[index,:]
tr_b = tr_b[index,:]
tr_y = tr_y[index]

valsamples=list(product(va_acc.index,rep_lan.index))
va_y = np.array([int(va_acc['Label'][sample[0]]!=rep_lan['Label'][sample[1]]) for sample in valsamples])
va_a = np.array([np.array(va_acc.loc[sample[0]]) for sample in valsamples])[:,1:401]
va_b = np.array([np.array(rep_lan.loc[sample[1]]) for sample in valsamples])[:,1:401]

print ("Data Load Complete")

shared_network = create_shared_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

processed_a = shared_network(input_a)
processed_b = shared_network(input_b)

distance = Lambda(euclidean_dist,output_shape=output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[custom_accuracy])

model.fit([tr_a,tr_b],tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([va_a,va_b],va_y))

tr_pred = model.predict([tr_a,tr_b])
tr_accu = compute_accuracy(tr_y,tr_pred)
va_pred = model.predict([va_a,va_b])
va_accu = compute_accuracy(va_y,va_pred)

model.save('/home/aditya/Desktop/accent_identification2/siamese_network/model/siamese01.h5')

print('* Accuracy on training set: %0.2f%%' % (100 * tr_accu))
print('* Accuracy on val set: %0.2f%%' % (100 * va_accu))

valdnsamp=[(sample[0],rep_lan['Label'][sample[1]]) for sample in valsamples]
valdnuniq=list(set(valdnsamp))
valdnuniq.sort()

final=np.zeros((va_acc.shape[0],len(levels)))
final=pd.DataFrame(final)
final.index=va_acc.index
final.columns=levels

def myfunction(tup):
    tempindex = [i for i, x in enumerate(valdnsamp) if x == tup]
    temppred = [va_pred[j] for j in tempindex]
    temppred = np.min(np.array(temppred))
    final[tup[1]][tup[0]] = temppred

supress=map(myfunction,valdnuniq)

fi_accu=float(np.sum(np.array(final.idxmin(1)==va_acc['Label'])))/(va_acc.shape[0])
print('* Final Accuracy: %0.2f%%' % (100 * fi_accu))