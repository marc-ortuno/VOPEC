import wave
import numpy as np
import librosa 

class Waveform:
    def __init__(self,path= None, signal=None, sample_rate = 44100, filename= None):
        """
        Instantiate an object of this class. This loads the audio into a (T,) np.array: self.waveform.
            Args:
                path (str): Path to a sound file (eg .wav or .mp3).
                sample_rate (int): Sample rate used to load the sound file.
                OR:
                signal (np.array, [T, ]): Waveform signal.
                sample_rate (int): Sample rate of the waveform.
        """
        if signal is None:
            assert isinstance(path, str), "The path argument to the constructor must be a string."

        if path is None:
            assert isinstance(signal, np.ndarray), "The signal argument to the constructor must be a np.array."
            assert len(signal.shape) == 1, "The signal argument to the constructor must be a 1D [T, ] array."

        if (signal is not None) and (path is not None):
            raise ValueError("Cannot give both a path to a sound file and a signal. Take your pick!")

        assert isinstance(sample_rate, int), "The sample_rate argument to the constructor must be an integer."

        if path is not None:
            self._waveform = librosa.core.load(path, sr=sample_rate)[0]
        else:
            self._waveform = signal


        self._sample_rate = sample_rate
        self._filename = filename
        self._path = path
    
    @property
    def waveform(self):
        """Properties written in this way prevent users to assign to self.waveform"""
        return self._waveform

    @property
    def sample_rate(self):
        """Properties written in this way prevent users to assign to self.sample_rate"""
        return self._sample_rate

    def magnitude_spectrum(self, n_fft=512, hop_length=128):
        """Compute the STFT of self.waveform. This is used for further spectral analysis.
        Args:
            n_fft_seconds (float): Length of the FFT window in seconds.
            hop_length_seconds (float): How much the window shifts for every timestep, in seconds.
        Returns:
            np.array, [n_fft / 2 + 1, T / hop_length]: The magnitude spectrogram
        """
        mag_spectrum, _ = librosa.core.spectrum._spectrogram(self.waveform, n_fft=n_fft, hop_length=hop_length)
        return mag_spectrum