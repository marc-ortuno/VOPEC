import numpy as np
import librosa
from utils import plot_librosa_spectrum
import scipy
import scipy.signal as sc
import madmom



def activity_detection(func_type,signal,sample_rate,buffer_len,previous_ODF):
    return {
        1: lambda: activity_detection_1(signal,sample_rate,buffer_len,previous_ODF),
        2: lambda: activity_detection_2(signal,sample_rate,buffer_len,previous_ODF),
    }[func_type]()
    
def activity_detection_1(signal,sample_rate,buffer_len,previous_ODF):
    """
    Activity-Detection interface
    :param signal: Signal
    :param sample_rate: Int
    :output flag: Boolean (onset/offset)
    :output signal_hfc: High Frequency Content Function
    """
    flag = False

    #Compute FFT and HFCv2
    signal_fft = np.fft.fft(signal,signal.size)
    ODF = hfc(signal_fft)

    # Generally, an adaptive threshold is computed as a
    # smoothed version of the detection function.

    l = 0.5
    a = 0.5
    m = 128
    b = 2

    last_ODF = previous_ODF[-1:]
    extended_ODF = np.append(np.array(last_ODF),ODF)
    
    threshold = np.zeros(len(ODF))
    if len(extended_ODF) > buffer_len + m:
        for i in range(buffer_len,buffer_len* 2 ,1):
            threshold[i - buffer_len] = (l * np.median(extended_ODF[i-m:i+b]) + a * np.mean(extended_ODF[i-m:i+b])) 
    
    #Peak Picking
    # values = sum(i >= threshold for i in ODF)
    # print(values)

    if any(ODF) < any(threshold):
        flag = True
    else:
        flag = False

    return flag, ODF,threshold

def activity_detection_2(signal,sample_rate,buffer_len,previous_ODF):
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
    ODF = hfc2(signal_fft)

    #Since we want to avoid the false peaks and the onset-detection is real-time, we use calculate
    #the thresholds using a slight variation of the median/mean function for each frame.
    #https://asp-eurasipjournals.springeropen.com/track/pdf/10.1186/1687-6180-2011-68.pdf

    l = 1
    a = 2
    m = 7
    
    m_ODF = previous_ODF[-m:]
    m_ODF.append(ODF)
    nm_ODF = m_ODF
    if len(m_ODF) >= m:
        threshold = (l * np.median(nm_ODF) + a * np.mean(nm_ODF)) 

    


    #Peak Picking
    # values = sum(i >= threshold for i in ODF)
    # print(values)

    if ODF > threshold and ODF > 10:
        flag = True
    else:
        flag = False

    return flag, ODF,threshold




# Onset Detection Functions (no mean)
def hfc(fft):

    hfc = np.abs(np.power(fft, 2)) * np.arange(1, fft.size + 1)
    return hfc



# Onset Detection Functions (mean)
def hfc2(fft):

    hfc = np.sum(np.abs(np.power(fft, 2)) * np.arange(1, fft.size + 1))
    return hfc




