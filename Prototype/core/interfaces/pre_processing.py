import numpy as np
import scipy.signal as scipy_signal
from matplotlib import pyplot as plt
from utils import plot_fft,plot_audio

def pre_processing(signal,fs):
    """
    Pre-processing interface
    :param signal: Signal
    :output processed_signal : Signal
    """
    
    # norm = np.linalg.norm(signal)
    # normal_array = signal/norm
    # processed_signal = normal_array

    # plot_audio(processed_signal,fs)
    
    n = signal.size 
    fhat = np.fft.fft(signal,n) # Compute FFT
    PSD = fhat*np.conj(fhat)/n # Power Spectrum

    #Use the PSD to filter noise
    indices = PSD > 0.03
    PSDclean = PSD * indices
    fhat = indices * fhat
    denoised_signal = np.fft.ifft(fhat)

    # plot_fft(signal,denoised_signal,fs,PSD,PSDclean,n)
    
    #High Pass Filter
    hp_filter = scipy_signal.butter(10, 500, 'hp', fs=fs,output='sos')
    filtered_signal = scipy_signal.sosfilt(hp_filter, np.abs(denoised_signal))
    
    
    return filtered_signal   