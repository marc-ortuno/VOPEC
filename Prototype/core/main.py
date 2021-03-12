import os
import numpy as np
from dataset import get_dataset
from utils import plot_audio, plot_spectrum, plot_odf
from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from models import Waveform
from matplotlib import pyplot as plt
from test import evaluate_system
import librosa

#Load dataset to train ML
data = get_dataset(sound="Kick",microphone="2") 
groundtruth = []
predicted = []
# wave_data = data[5]
# plot_audio(wave_data)

# for sound in data:
audio_sample_rate = 44100
# spectrum = audio.magnitude_spectrum()

signal = Waveform(path="./data/AFRP2.wav").waveform
data_type = signal.dtype
signal_duration = signal.size/audio_sample_rate;


buffer_size = 512
hop_size = 128
duration = 0
w_duration = hop_size*signal_duration/signal.size
features = []
last_onset = False
hfc = []
# print(signal[7552:8064].size)
print(signal.size)
for i in range(0,signal.size,hop_size):
    #windowing signal
    w_signal = signal[i:(i+buffer_size)]

    #Pre-processing signal
    n_signal = pre_processing(w_signal)

    #Detect onset
    onset, odf = activity_detection(n_signal,audio_sample_rate,buffer_size,hop_size)
    hfc.append(odf)
    # Only calculated when the onset is detected
    if onset:
        # print("Onset at:", duration )
        w_features = feature_extraction(n_signal)
        features.append(w_features)

    #Offset detected
    if (last_onset is True and onset is False) or (i+hop_size >= signal.size and onset is True):
        predicted.append(classificator(features))
    
    last_onset = onset
    duration = duration + w_duration

# evaluate_system(groundtruth,predicted)
# print(hfc)
plot_odf(hfc,audio_sample_rate,signal)