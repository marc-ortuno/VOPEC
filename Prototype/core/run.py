from app import main
from models import Waveform
from utils import load_groundtruth

audio = Waveform(path="./data/Freestyle_OBM.wav")
groundtruth = load_groundtruth('./data/Freestyle_OBM.csv')

_ = main(audio,groundtruth)