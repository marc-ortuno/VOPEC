import numpy as np
import librosa
from utils import plot_librosa_spectrum
import scipy
import scipy.signal as sc
import madmom

def activity_detection(signal,sample_rate,n_fft,hop_size):
    """
    Activity-Detection interface
    :param signal: Signal
    :param sample_rate: Int
    :param n_fft: Int
    :param hop_size: Int
    :output flag: Boolean
    """
    flag = False

    #Compute FFT and HFCv2
    signal_fft = np.fft.fft(signal,signal.size)
    signal_hfc = hfc(signal_fft)
    # print(signal_hfcv2)


    #Peak Picking -> Static threshold
    th = 30000

    if th < signal_hfc:
        flag = True
    else:
        flag = False

    return flag, signal_hfc



# Onset Detection Functions
def hfc(fft):

    hfc = np.sum(np.abs(np.power(fft, 2)) * np.arange(1, fft.size + 1))
    return hfc




