import numpy as np
import librosa
from utils import plot_librosa_spectrum
import scipy
import scipy.signal as sc
import madmom



def activity_detection(func_type,signal,sample_rate,buffer_len,previous_ODF,highest_peak):
    return {
        1: lambda: activity_detection_1(signal,sample_rate,buffer_len,previous_ODF,highest_peak),
    }[func_type]()
    


def activity_detection_1(signal,sample_rate,buffer_len,previous_ODF,highest_peak):
    """
    Activity-Detection interface
    :param signal: Signal
    :param sample_rate: Int
    :output flag: Boolean (onset/offset)
    :output signal_hfc: High Frequency Content Function
    """

    flag = False
    threshold = 0

    #Compute FFT and HFCv2
    signal_fft = np.fft.fft(signal,signal.size)
    ODF = hfc(signal_fft)

    if ODF >= np.mean(highest_peak):
        highest_peak.append(ODF)


    #Since we want to avoid the false peaks and the onset-detection is real-time, we use calculate
    #the thresholds using a slight variation of the median/mean function for each frame.
    #https://asp-eurasipjournals.springeropen.com/track/pdf/10.1186/1687-6180-2011-68.pdf

    l = 1
    a = 2
    m = 7
    
    N = np.mean(highest_peak) * 0.03
    # print(N)
    m_ODF = previous_ODF[-m:]
    m_ODF.append(ODF)
    nm_ODF = m_ODF

    threshold = (l * np.median(nm_ODF) + a * np.mean(nm_ODF)) + N



    #Peak Picking
    # values = sum(i >= threshold for i in ODF)
    # print(values)

    if ODF > threshold:
        flag = True
    else:
        flag = False

    return flag, ODF,threshold,highest_peak

def hfc(fft):

    hfc = np.sum(np.abs(np.power(fft, 2)) * np.arange(1, fft.size + 1))
    return hfc




