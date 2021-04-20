from models import Waveform
from simulation import main

audio = Waveform(path="./data/Freestyle_MOB.wav")
main(audio)