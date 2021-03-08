from matplotlib import pyplot as plt
import numpy as np

"""
This function plot an Audio object.
"""
def plot_audio(wave_file):
    plt.figure(1)
    
    plot_a = plt.subplot(211)
    plot_a.plot(wave_file.audio)
    plot_a.set_title(wave_file.filename)
    plot_a.set_xlabel('Sample rate * time')
    plot_a.set_ylabel('Energy')
    
    plot_b = plt.subplot(212)
    plot_b.specgram(wave_file.audio, NFFT=1024, Fs=wave_file.rate, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')
    
    plt.show()