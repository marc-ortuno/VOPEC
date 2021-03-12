
# From surfboard import Waveform Class: https://github.com/novoic/surfboard
import numpy as np
import librosa 

class Waveform:
    def __init__(self,path= None, signal=None, sample_rate = 44100, filename= None,class_type=None):
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

        assert self.waveform.shape[0] > 1, "Your waveform must have more than one element."

        self._sample_rate = sample_rate
        self._filename = filename
        self._class_type = class_type
        self._duration = librosa.get_duration(y=self._waveform,sr=self._sample_rate)
        self._path = path
    
    @property
    def waveform(self):
        """Properties written in this way prevent users to assign to self.waveform"""
        return self._waveform

    @property
    def sample_rate(self):
        """Properties written in this way prevent users to assign to self.sample_rate"""
        return self._sample_rate

    @property
    def filename(self):
        """Properties written in this way prevent users to assign to self.filename"""
        return self._filename

    @property
    def class_type(self):
        """Properties written in this way prevent users to assign to self.class_type"""
        return self._class_type

    @property
    def duration(self):
        """Properties written in this way prevent users to assign to self.duration"""
        return self._duration

    @property
    def path(self):
        """Properties written in this way prevent users to assign to self.path"""
        return self._path