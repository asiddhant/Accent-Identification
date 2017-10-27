import librosa
import numpy as numpy
import pandas as pd 
import os

filepath='/home/aditya/Desktop/MFCCs/accfiles/'
# for filename in os.listdir(filepath):
#     y, sr = librosa.load('data/' + filename)
#     print filename, librosa.feature.mfcc(y=y, sr=sr).shape
filename=os.listdir(filepath)
audio,sr=librosa.load(filepath+filename[0])
mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
mfcd=librosa.feature.delta(mfcc)
mfdd=librosa.feature.delta(mfcd)
fv=