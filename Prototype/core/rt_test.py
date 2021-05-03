#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""

import numpy as np
import sounddevice as sd
from app import main,process,init,init_activity_detection,init_classificator,init_feature_extraction,init_pre_processing
from utils import plot_audio
import pickle
import tempfile
from scipy.io.wavfile import write

#Import model
filename = './app/finalized_model.sav'
knn_model = pickle.load(open(filename, 'rb'))

signal_original = [] 
signal_processed = []

duration = 10

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    
    # Fancy indexing with mapping creates a (necessary!) copy:
    input_buffer = indata.flatten()
    prediction = process(input_buffer, signal_processed)[6]
    if prediction != "":
        print(prediction)
    # signal_processed.extend(n_signal)
    signal_original.extend(input_buffer)

           

try:

    global running
    running = True

    device_info = sd.query_devices(sd.default.device, 'input')
    samplerate = device_info['default_samplerate']

    buffer_size = 512

    init_pre_processing()
    init_activity_detection(func_type=2)
    init_feature_extraction()
    init_classificator(knn_model = knn_model)
    init(samplerate, buffer_size)

    stream = sd.InputStream(
        device=sd.default.device, channels=1, dtype='float32',latency= 'low',
        samplerate=samplerate, callback=audio_callback, blocksize=buffer_size)

    with stream:
        sd.sleep(int(duration * 1000))

except KeyboardInterrupt:
        sd.play(signal_original)
        filename = tempfile.mktemp(prefix='./demo/realtime_demo_',
                                        suffix='.wav', dir='')
        write(filename, int(samplerate), np.asarray(signal_original))
        plot_audio(signal_original,signal_processed,44100)
        sys.exit()

