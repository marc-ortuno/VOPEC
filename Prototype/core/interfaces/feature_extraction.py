import numpy as np


def feature_extraction(signal):
    """
    Feature extraction interface
    :param signal: Signal
    :output features: Array of features
    
    Features are extracted from the incoming audio signal when an onset is detected.
    """
    features = []
    for sample in signal:
        if sample > 0.1:
            features.append(1)
        else:
            features.append(0)
    return features   