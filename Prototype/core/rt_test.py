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
from utils import plot_audio
import pickle
import tempfile
from scipy.io.wavfile import write
from models import Waveform

# Import model


filename = './app/finalized_model_v2.sav'
knn_model = pickle.load(open(filename, 'rb'))

signal_original = []
signal_processed = []

duration = 10

# Playback sounds
Kick = Waveform(path="data/Kick.wav")
Snare = Waveform(path="./data/Snare.wav")
HH = Waveform(path="./data/HH.wav")


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""

    # Fancy indexing with mapping creates a (necessary!) copy:
    input_buffer = indata.flatten()
    n_signal, _, _, _, _, _, prediction = process(input_buffer, signal_processed)
    if prediction != "":
        print(prediction)
        # play_sound(prediction)
    signal_processed.extend(n_signal)
    signal_original.extend(input_buffer)


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
    samplerate = device_info['default_samplerate']

    buffer_size = 512

    init_pre_processing()
    init_activity_detection(func_type=1)
    init_feature_extraction(n_mfcc_arg=10)
    init_classificator(knn_model=knn_model)
    init(samplerate, buffer_size)

    stream = sd.InputStream(
        device=sd.default.device, channels=1, dtype='float32', latency='low',
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
    sys.exit()
