#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.
Matplotlib and NumPy have to be installed.
"""
print("Initialising... please wait")

import tkinter as tk
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

# GUI
window = tk.Tk()
window.geometry("400x400")
window.title("VOPEC")
selected = ""
prediction_label = tk.Label(window, text="", font='Helvetica 18 bold')
# Playback sounds
Kick = Waveform(path="data/Kick.wav")
Snare = Waveform(path="./data/Snare.wav")
HH = Waveform(path="./data/HH.wav")

buffer_size = 512
sample_rate = 44100

device = None


def audio_callback(indata, frames, time, status):
    global onset_location
    """This is called (from a separate thread) for each audio block."""
    # Fancy indexing with mapping creates a (necessary!) copy:
    input_buffer = indata.flatten()
    n_signal, _, hfc, _, onsets, th, prediction = process(input_buffer, signal_processed)
    if prediction != "":
        prediction_label.config(text=prediction)
        # play_sound(prediction)
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


def on_closing():
    window.destroy()
    # sd.play(signal_original)
    filename = tempfile.mktemp(prefix='./demo/realtime_demo_',
                               suffix='.wav', dir='')
    write(filename, sample_rate, np.asarray(signal_original))
    plot_audio(np.array(signal_original), np.array(signal_processed), sample_rate)
    plot_odf(filename, np.array(signal_original), np.array(signal_processed), sample_rate, np.array(onset_location),
             np.array(total_odf), np.array(total_th))
    sys.exit()

def select_device_and_run(event):
    myLabel = tk.Label(window, text="Device: " + variable.get()).pack()
    w.destroy()
    device_label.destroy()
    prediction_label.place(relx=.5, rely=.5, anchor="center")

    # set device
    device_info = sd.query_devices(get_host_id(variable.get()), 'input')
    device = get_host_id(variable.get())
    sample_rate = device_info['default_samplerate']
    channels = 1

    init_pre_processing()
    init_activity_detection()
    init_feature_extraction(func_type="mfcc", n_mfcc_arg=20, norm_file=normalization_values)
    init_classificator(knn_model=knn_model)
    init(sample_rate, buffer_size)

    stream = sd.InputStream(
        device=device, channels=channels, dtype='float32', latency='high',
        samplerate=sample_rate, callback=audio_callback, blocksize=buffer_size)

    window.protocol("WM_DELETE_WINDOW", on_closing)

    with stream:
        print("Recording***")
        window.mainloop()


def get_host_id(item):
    """
    get id of the device
    """
    return int(item.split(" ")[0])


try:
    global running
    running = True

    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append(str(index) + " " + device.get('name'))
    OPTIONS = devices

    variable = tk.StringVar(window)
    variable.set(OPTIONS[len(OPTIONS) - 1])  # default value

    device_label = tk.Label(window, text="Select device")
    device_label.place(relx=.5, rely=.4, anchor="center")
    w = tk.OptionMenu(window, variable, *OPTIONS, command=select_device_and_run)
    w.place(relx=.5, rely=.5, anchor="center")

    window.mainloop()

except ValueError:
    print("An exception occurred")
