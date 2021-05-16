import numpy as np
import librosa
from utils import plot_librosa_spectrum
import scipy
import scipy.signal as sc
import madmom


def activity_detection(func_type, signal, sample_rate, buffer_len, previous_ODF, highest_peak):
    return {
        1: lambda: activity_detection_1(signal, sample_rate, buffer_len, previous_ODF, highest_peak),
    }[func_type]()


def activity_detection_1(signal, sample_rate, buffer_len, previous_ODF, highest_peak):
    """
    Activity-Detection interface
    :param signal: Sub-band Signal
    :param sample_rate: Int
    :output flag: Boolean (onset/offset)
    :output signal_hfc: High Frequency Content Function
    """

    flag = False
    threshold = np.zeros(len(signal))
    # Compute FFT and HFCv2
    ODF = hfc(signal)

    peak_indices = np.where(ODF > np.mean(highest_peak))

    if peak_indices[0].size > 0:
        highest_peak = np.vstack([highest_peak, ODF])

    # Since we want to avoid the false peaks and the onset-detection is real-time, we use calculate
    # the thresholds using a slight variation of the median/mean function for each frame.
    # https://asp-eurasipjournals.springeropen.com/track/pdf/10.1186/1687-6180-2011-68.pdf

    l = 0.3
    a = 0.5
    d = 0.03

    m = 7

    band_onset = 0

    for i in range(0, len(ODF)):
        if highest_peak.ndim >= 2:
            N = np.mean(highest_peak[:, i]) * d
        else:
            N = np.mean(highest_peak) * d


        # print(N)
        if previous_ODF.ndim >= 2:
            m_ODF = previous_ODF[-m:, i]
            m_ODF = np.append(m_ODF, ODF[i])
            nm_ODF = m_ODF
            threshold[i] = (l * np.median(nm_ODF) + a * np.mean(nm_ODF)) + N
        else:
            threshold[i] = N

        if ODF[i] > threshold[i]:
            band_onset += 1


    # Peak Picking
    # values = sum(i >= threshold for i in ODF)
    # print(values)
    bands_threshold = int((len(signal) // 1.75))
    if band_onset >= bands_threshold:
        flag = True
    else:
        flag = False

    return flag, ODF, threshold, highest_peak


def hfc(fft):
    hfc = np.zeros(len(fft))
    for i in range(0, len(fft)):
        fft_band = fft[i]
        hfc[i] = np.sum(np.abs(np.power(fft_band, 2)) * np.arange(1, fft_band.size + 1))
    return hfc
