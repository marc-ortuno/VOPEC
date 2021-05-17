#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.
Matplotlib and NumPy have to be installed.
"""
print("Initialising... please wait")

import numpy as np
import sys
import sounddevice as sd
from app import main, process, init, init_activity_detection, init_classificator, init_feature_extraction, \
    init_pre_processing
from utils import plot_audio, plot_odf
import pickle
import tempfile
from scipy.io.wavfile import write
from models import Waveform
import pandas as pd

"""
This script executes a real time demonstration of the prototype. It uses sounddevice to obtain streams of a microphone 
input. On each callback we execute de process of the prototype.
"""

# Import model
filename = './app/finalized_model_mfccs.sav'
knn_model = pickle.load(open(filename, 'rb'))

model_normalization = './app/model_normalization_mfccs.csv'
normalization_values = pd.read_csv(model_normalization)

signal_original = []
signal_processed = []
total_odf = []
total_th = []
onset_location = []
duration = 10

# Playback sounds
Kick = Waveform(path="data/Kick.wav")
Snare = Waveform(path="./data/Snare.wav")
HH = Waveform(path="./data/HH.wav")

buffer_size = 512


def audio_callback(indata, frames, time, status):
    global onset_location
    """This is called (from a separate thread) for each audio block."""

    # Fancy indexing with mapping creates a (necessary!) copy:
    input_buffer = indata.flatten()
    n_signal, _, hfc, _, onsets, th, prediction = process(input_buffer, signal_processed)
    if prediction != "":
        print(prediction)
        play_sound(prediction)
    signal_processed.extend(n_signal)
    signal_original.extend(input_buffer)
    total_odf.extend([np.sum(hfc)] * buffer_size)
    total_th.extend([np.sum(th)] * buffer_size)
    onset_location = onsets


def play_sound(class_tag):
    if class_tag == "Kick":
        sd.play(Kick.waveform)
    elif class_tag == "Snare":
        sd.play(Snare.waveform)
    else:
        sd.play(HH.waveform)


try:
    global running
    running = True

    device_info = sd.query_devices(sd.default.device, 'input')
    device = 5
    samplerate = device_info['default_samplerate']
    # print(sd.query_devices())

    init_pre_processing()
    init_activity_detection()
    init_feature_extraction(func_type="mfcc", n_mfcc_arg=20, norm_file=normalization_values)
    init_classificator(knn_model=knn_model)
    init(samplerate, buffer_size)

    stream = sd.InputStream(
        device=device, channels=1, dtype='float32', latency='high',
        samplerate=samplerate, callback=audio_callback, blocksize=buffer_size)

    with stream:
        print("Recording***")
        sd.sleep(int(duration * 1000))

except KeyboardInterrupt:
    sd.play(signal_original)
    filename = tempfile.mktemp(prefix='./demo/realtime_demo_',
                               suffix='.wav', dir='')
    write(filename, int(samplerate), np.asarray(signal_original))
    plot_audio(signal_original, signal_processed, 44100)
    plot_odf(filename, np.array(signal_original), signal_processed, 44100, onset_location, total_odf, total_th)
    sys.exit()
