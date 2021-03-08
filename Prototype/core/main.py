from dataset import Reader
from utils import plot_audio
from interfaces import pre_processing
import os


reader = Reader()
data = reader.get_dataset("Snare","2") 
wave_file = data[5]
plot_audio(wave_file)

signal = pre_processing(wave_file)
print(signal)

