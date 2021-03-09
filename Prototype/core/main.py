import os
import numpy as np
import wave
from dataset import get_dataset
from utils import plot_audio
from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from models import Waveform
from matplotlib import pyplot as plt

#Load dataset to train ML
data = get_dataset("Snare","2") 
wave_data = data[5]
# plot_audio(wave_data)



audio = wave.open("./data/AFRP2.wav","rb")
nframes = audio.getnframes()
framerate = audio.getframerate()
signal = audio.readframes(nframes)
signal_duration = nframes/framerate;
audio.close()
signal = np.frombuffer(signal, dtype=np.short)

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