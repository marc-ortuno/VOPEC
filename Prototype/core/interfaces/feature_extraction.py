import numpy as np
import librosa.feature as feature
import librosa


def feature_extraction(func_type, signal, sr, n_mfcc, buffer_len, normalization_values):
    return {
        "mfcc": lambda: feature_extraction_mfcc(signal, sr, n_mfcc, buffer_len, normalization_values),
        "all": lambda: feature_extraction_all(signal, sr, n_mfcc, buffer_len, normalization_values),
    }[func_type]()


def feature_extraction_mfcc(signal, sr, n_mfcc, buffer_len, normalization_values=[]):
    """
    Feature extraction interface
    :param signal: Signal
    :output features: Array of features

    Features are extracted from the incoming audio signal when an onset is detected.
    """
    signal = np.array(signal)
    features = []
    if signal.size != 0:
        mfcc = feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=int(512 * 2), hop_length=int(128 * 2))
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        if len(normalization_values) > 1:
            features.extend(normalize(mfcc_mean, normalization_values[['mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3',
                                                                       'mfcc_mean_4', 'mfcc_mean_5', 'mfcc_mean_6',
                                                                       'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9',
                                                                       'mfcc_mean_10', 'mfcc_mean_11', 'mfcc_mean_12',
                                                                       'mfcc_mean_13', 'mfcc_mean_14', 'mfcc_mean_15',
                                                                       'mfcc_mean_16', 'mfcc_mean_17', 'mfcc_mean_18',
                                                                       'mfcc_mean_19', 'mfcc_mean_20']]))

            features.extend(normalize(mfcc_std, normalization_values[['mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3',
                                                                      'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6',
                                                                      'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9',
                                                                      'mfcc_std_10', 'mfcc_std_11', 'mfcc_std_12',
                                                                      'mfcc_std_13', 'mfcc_std_14', 'mfcc_std_15',
                                                                      'mfcc_std_16', 'mfcc_std_17', 'mfcc_std_18',
                                                                      'mfcc_std_19', 'mfcc_std_20']]))
        else:
            features.extend(mfcc_mean)
            features.extend(mfcc_std)

        # features.extend(mfcc_mean)

    return features


def feature_extraction_all(signal, sr, n_mfcc, buffer_len, normalization_values):
    """
    Feature extraction interface
    :param signal: Signal
    :param sr: Signal
    :param n_mfcc: Signal
    :param buffer_len: Signal
    :param normalization_values: normalization values of the dataset
    :output features: Array of features

    Features are extracted from the incoming audio signal when an onset is detected.
    """
    features = []
    signal = np.array(signal)

    if signal.size != 0:
        S, phase = librosa.magphase(librosa.stft(y=signal, n_fft=buffer_len, hop_length=int(buffer_len / 4)))

        # Mel Frequency cepstral coefficients
        mfcc = feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=int(512 * 2), hop_length=int(128 * 2))
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # RMS
        rms = feature.rms(S=S, frame_length=buffer_len, hop_length=int(buffer_len / 4))
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

        # Spectral Centroid
        spectral_centroid = feature.spectral_centroid(S=S, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
        spectral_centroid_std = np.std(spectral_centroid, axis=1)

        # Rolloff
        spectral_rolloff = feature.spectral_rolloff(S=S, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff, axis=1)
        spectral_rolloff_std = np.std(spectral_rolloff, axis=1)

        # Bandwidth
        spectral_bandwidth = feature.spectral_bandwidth(S=S, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth, axis=1)
        spectral_bandwidth_std = np.std(spectral_bandwidth, axis=1)

        # Contrast
        spectral_contrast = feature.spectral_contrast(S=S, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        spectral_contrast_std = np.std(spectral_contrast, axis=1)

        # Flatness
        spectral_flatness = feature.spectral_flatness(S=S)
        spectral_flatness_mean = np.mean(spectral_flatness, axis=1)
        spectral_flatness_std = np.std(spectral_flatness, axis=1)

        if len(normalization_values) > 1:
            # Duration
            features.append(normalize(len(signal), normalization_values['duration']))

            features.extend(normalize(mfcc_mean, normalization_values[['mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3',
                                                                       'mfcc_mean_4', 'mfcc_mean_5', 'mfcc_mean_6',
                                                                       'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9',
                                                                       'mfcc_mean_10']]))

            features.extend(normalize(mfcc_std, normalization_values[['mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3',
                                                                      'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6',
                                                                      'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9',
                                                                      'mfcc_std_10']]))

            features.extend(normalize(rms_mean, normalization_values['rms_mean']))

            features.extend(normalize(rms_std, normalization_values['rms_std']))

            features.extend(normalize(spectral_centroid_mean, normalization_values['spectral_centroid_mean']))

            features.extend(normalize(spectral_centroid_std, normalization_values['spectral_centroid_std']))

            features.extend(normalize(spectral_rolloff_mean, normalization_values['spectral_rolloff_mean']))

            features.extend(normalize(spectral_rolloff_std, normalization_values['spectral_rolloff_std']))

            features.extend(normalize(spectral_bandwidth_mean, normalization_values['spectral_bandwidth_mean']))

            features.extend(normalize(spectral_bandwidth_std, normalization_values['spectral_bandwidth_std']))

            features.extend(normalize(spectral_contrast_mean, normalization_values[['spectral_contrast_mean_1',
                                                                                    'spectral_contrast_mean_2',
                                                                                    'spectral_contrast_mean_3',
                                                                                    'spectral_contrast_mean_4',
                                                                                    'spectral_contrast_mean_5',
                                                                                    'spectral_contrast_mean_6',
                                                                                    'spectral_contrast_mean_7']]))

            features.extend(normalize(spectral_contrast_std, normalization_values[['spectral_contrast_std_1',
                                                                                   'spectral_contrast_std_2',
                                                                                   'spectral_contrast_std_3',
                                                                                   'spectral_contrast_std_4',
                                                                                   'spectral_contrast_std_5',
                                                                                   'spectral_contrast_std_6',
                                                                                   'spectral_contrast_std_7']]))

            features.extend(normalize(spectral_flatness_mean, normalization_values['spectral_flatness_mean']))

            features.extend(normalize(spectral_flatness_std, normalization_values['spectral_flatness_std']))
        else:
            features.append(len(signal))
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            features.extend(rms_mean)
            features.extend(rms_std)
            features.extend(spectral_centroid_mean)
            features.extend(spectral_centroid_std)
            features.extend(spectral_rolloff_mean)
            features.extend(spectral_rolloff_std)
            features.extend(spectral_bandwidth_mean)
            features.extend(spectral_bandwidth_std)
            features.extend(spectral_contrast_mean)
            features.extend(spectral_contrast_std)
            features.extend(spectral_flatness_mean)
            features.extend(spectral_flatness_std)

        features = np.array(features)
    return features


# Min-Max Scaling.
def normalize(data, normalization_values):
    ssd = normalization_values.values[0]
    data_norm = (data - normalization_values.values[1]) / (normalization_values.values[0] -
                                                           normalization_values.values[1])
    return data_norm
