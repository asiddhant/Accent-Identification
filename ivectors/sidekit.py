# Python Script for I-Vector Extraction
# Aditya Siddhant (siddhantaditya01@gmail.com)

# Required Libraries
import os
import sidekit
import scipy
import numpy as np 
import pandas as pd 


# Choose the directory containing the folders for different accented speech data.
# Requirements: The subfolder names must be class identifiers and Subfolders must contain .wav files only.

inputpath = "/home/aditya/Desktop/MFCCs"

show_list = []
channel_list = []
for filename in os.listdir(inputpath+"/accfiles"):
	show_list += [os.path.splitext(filename)[0]]
	channel_list += [0]

extractor = sidekit.FeaturesExtractor(audio_filename_structure=inputpath+"/accfiles/"+"{}.wav",
	                                  feature_filename_structure=inputpath+"/features/"+"{}.hd5",
	                                  sampling_frequency=8000,
	                                  lower_frequency=200,
	                                  higher_frequency=3800,
	                                  filter_bank="log",
	                                  filter_bank_size=24,
	                                  window_size=0.02,
	                                  shift=0.01,
	                                  ceps_number=20,
	                                  vad="snr",
	                                  snr=40,
	                                  pre_emphasis=0.97,
	                                  save_param=["energy", "cep"],
	                                  keep_all_features=True)

extractor.save_list(show_list=show_list,
                    channel_list=channel_list,
                    num_thread=6)



