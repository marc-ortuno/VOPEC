import wave
import numpy as np

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
            self._waveform = wave.open(path,"rb")
            self._sample_rate = self._waveform.getframerate()
        else:
            self._waveform = signal
            self._sample_rate = sample_rate



        self._filename = filename
        self._path = path