import numpy as np


def pre_processing(signal):
    """
    Pre-processing interface
    :param signal: Signal
    :output processed_signal : Signal
    """
    norm = np.linalg.norm(signal)
    normal_array = signal/norm
    processed_signal = normal_array
    
    return processed_signal   