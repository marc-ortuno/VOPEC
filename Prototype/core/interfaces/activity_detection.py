import numpy as np


def activity_detection(signal):
    """
    Activity-Detection interface
    :param signal: Signal
    :output onset: Boolean
    """
    onset = False
    total = np.abs(np.sum(signal))
    if total > 4:
        onset = True
    else:
        onset = False
    return onset,total  