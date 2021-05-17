from pyfilterbank import FractionalOctaveFilterbank
import numpy as np
import scipy.signal as scipy_signal
from matplotlib import pyplot as plt
from utils import plot_fft, plot_audio


def pre_processing(signal, fs, N):
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

    # High Pass Filter
    hp_fc = 500
    hp_filter = scipy_signal.butter(2, hp_fc, 'hp', fs=fs, output='sos')
    hp_filtered_signal = scipy_signal.sosfilt(hp_filter, np.abs(signal))


    # sub-band peak score detection
    # Several onset detection studies have found it useful to independently analyze information across different frequency
    # bands. In some cases this preprocessing is needed to satisfy
    # the needs of specific applications that require detection in individual sub-bands to complement global estimates;
    fft_signal = np.fft.fft(hp_filtered_signal, n)  # Compute FFT

    sub_band_signal = np.array(np.array_split(fft_signal, N))

    return sub_band_signal, hp_filtered_signal
