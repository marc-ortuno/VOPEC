import numpy as np
import librosa.feature as feature

def feature_extraction(signal,samp_freq,n_mfcc):
    """
    Feature extraction interface
    :param signal: Signal
    :output features: Array of features
    
    Features are extracted from the incoming audio signal when an onset is detected.
    """
    signal = np.array(signal)
    features = feature.mfcc(y=signal,sr=samp_freq,n_mfcc=n_mfcc,n_fft=1012, hop_length=256)
    features = np.mean(features,axis=1)
    return features