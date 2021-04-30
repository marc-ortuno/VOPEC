from pyfilterbank import FractionalOctaveFilterbank
import numpy as np
import scipy.signal as scipy_signal
from matplotlib import pyplot as plt
from utils import plot_fft,plot_audio

def pre_processing(signal,fs):
    """
    Pre-processing interface:
    The concept of preprocessing implies the transformation of
    the original signal in order to accentuate or attenuate various
    aspects of the signal according to their relevance to the task in
    hand. It is an optional step that derives its relevance 

    :param signal: Signal
    :output processed_signal : Signal
    """
    
    # norm = np.linalg.norm(signal)
    # normal_array = signal/norm
    # processed_signal = normal_array

    # plot_audio(processed_signal,fs)
    
    n = signal.size 


    # plot_fft(signal,denoised_signal,fs,PSD,PSDclean,n)


    #High Pass Filter
    hp_fc = 2500
    hp_filter = scipy_signal.butter(2, hp_fc, 'hp', fs=fs,output='sos')
    hp_filtered_signal = scipy_signal.sosfilt(hp_filter, np.abs(signal))

    #Spectral gating - D-noise
    fhat = np.fft.fft(hp_filtered_signal,n) # Compute FFT
    PSD = fhat*np.conj(fhat)/n # Power Spectrum
    indices = PSD > 0.001
    PSDclean = PSD * indices
    fhat = indices * fhat
    denoised_signal =np.fft.ifft(fhat)
    

    #Squaring the filtered signal to compute the energy.
    #This method discarded the negative part ofthe signal, which was useful to limit spurious detections after theattack

    squared_signal = np.square(hp_filtered_signal)
    
    #Low pass filter
    lp_fc = 25
    lp_filter = scipy_signal.butter(1, lp_fc, 'lp', fs=fs,output='sos')
    lp_filtered_signal = scipy_signal.sosfilt(lp_filter, np.abs(hp_filtered_signal))


    
    return hp_filtered_signal.real