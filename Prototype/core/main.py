import os
import numpy as np
from dataset import get_dataset
from utils import plot_audio
from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from models import Waveform
from matplotlib import pyplot as plt


#Load dataset to train ML
data = get_dataset("Snare","2") 
wave_data = data[5]
# plot_audio(wave_data)


audio_sample_rate = 44100
audio = Waveform(path="./data/AFRP2.wav",sample_rate=audio_sample_rate)
signal = audio.waveform
signal_duration = signal.size/audio_sample_rate;


buffer_size = 512
hop_size = 128
duration = 0
w_duration = hop_size*signal_duration/signal.size
total_signal = []

for i in range (0,signal.size,hop_size):
    w_signal = signal[i:(i+buffer_size)]
    n_signal = pre_processing(w_signal)
    # plot_audio(Waveform(signal=w_signal))
    onset,total = activity_detection(n_signal)
    total_signal.append(total)
    
    # Only calculated when the onset is detected
    
    if onset:
        print("Onset at:", duration )
        features = feature_extraction(n_signal)
    
    duration = duration + w_duration
    
    
plt.plot(total_signal)
plt.xlabel('t');
plt.ylabel('x(t)');
plt.show()