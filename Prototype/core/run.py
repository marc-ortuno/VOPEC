from app import main
from models import Waveform
from utils import load_groundtruth
from utils import plot_audio, plot_spectrum, plot_odf,plot_librosa_spectrum,plot_fft,plot_confusion_matrix,plot_evaluation_report
from evaluate import evaluate_system
import pickle

#Import model
filename = './app/finalized_model.sav'
knn_model = pickle.load(open(filename, 'rb'))

audio = Waveform(path="./data/Freestyle_JMP.wav")
groundtruth = load_groundtruth('./data/Freestyle_JMP.csv')

#Signal details
signal = audio.waveform
samp_freq = audio.sample_rate
name = audio.filename
buffer_len = 512
n_buffers = len(signal)//buffer_len

signal_proc, onset_location, total_hfc, predicted, _ = main(signal,samp_freq,n_buffers,buffer_len,knn_model,classificator_by_pass=False)

plot_odf(name,signal,signal_proc,samp_freq,onset_location,total_hfc)
plot_audio(signal,signal_proc,samp_freq)
report, cm = evaluate_system(groundtruth,predicted)

#EVALUATION METRICS PLOTS
plot_evaluation_report(report)
plot_confusion_matrix(cm)
