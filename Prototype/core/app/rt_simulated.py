import numpy as np
from dataset import get_dataset, dataset_analyzer
from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from utils import plot_audio, plot_spectrum, plot_odf,plot_librosa_spectrum,plot_fft,plot_confusion_matrix,plot_evaluation_report
from test import evaluate_system
from sklearn.metrics import classification_report
import pickle 

filename = './app/finalized_model.sav'
knn_model = pickle.load(open(filename, 'rb'))

# state variables
def init():

    # declare variables used in `process`
    # global
    global data
    global predicted 
    global features
    global n_fft
    global hop_size
    global last_onset 
    global hfc
    global n_mfcc
    global active_signal

    data = get_dataset(sound="Kick",microphone="2") 
    predicted = []

    n_fft = 256
    hop_size = 64
    features = []
    last_onset = False
    hfc = 0

    active_signal = []
    n_mfcc = 20

    return


# the process function!
def process(input_buffer, output_buffer, buffer_len,samp_freq,onset_location,t_audio_size):

    global last_onset
    global hfc
    global features
    global n_fft
    global hop_size
    global n_mfcc
    global active_signal

    features = []
    n_signal = pre_processing(input_buffer,samp_freq)

    #Detect onset
    onset,hfc = activity_detection(n_signal,samp_freq,n_fft,hop_size)
    # Only calculated when the onset is detected
    if onset:
        active_signal.extend(input_buffer) # Until we get an offset, the active sound is stored for later do feature extraction and classification.
        onset_location.extend(np.ones(buffer_len)) # Onset location for visual analysis
    else:
        onset_location.extend(np.zeros(buffer_len))

    #Offset detected
    if (last_onset is True and onset is False) or (len(active_signal) >= t_audio_size):
        features = feature_extraction(active_signal,samp_freq,n_mfcc)
        # predicted.append(classificator(features,knn_model))
        active_signal = [] #Clean active signal buffer
    
    last_onset = onset

    output_buffer = n_signal
    return output_buffer,features,hfc,predicted
    
def main(input_signal,groundtruth):
    init()
    signal = input_signal.waveform
    name = input_signal.filename
    samp_freq = input_signal.sample_rate
    # parameters
    buffer_len = 512
    n_buffers = len(signal)//buffer_len
    data_type = signal.dtype

    # allocate input and output buffers
    input_buffer = np.zeros(buffer_len, dtype=data_type)
    output_buffer = np.zeros(buffer_len, dtype=data_type)
    onset_location = []
    total_features = []
    total_hfc = []
    # simulate block based processing
    signal_proc = np.zeros(n_buffers*buffer_len, dtype=data_type)
    samp_freq = input_signal.sample_rate
    for k in range(n_buffers):

        # index the appropriate samples
        input_buffer = signal[k*buffer_len:(k+1)*buffer_len]
        output_buffer,features,hfc,predicted= process(input_buffer, output_buffer, buffer_len,samp_freq,onset_location,len(signal))
        signal_proc[k*buffer_len:(k+1)*buffer_len] = output_buffer
        total_features.extend(features)
        total_hfc.extend([hfc]*output_buffer.size)


    plot_odf(name,signal,signal_proc,samp_freq,onset_location,total_hfc)
    # print(predicted)
    print(len(total_features))
    # plot_audio(input_signal,signal_proc,samp_freq)
    # report, cm = evaluate_system(groundtruth,predicted)

    # #EVALUATION METRICS PLOTS
    # plot_evaluation_report(report)
    # plot_confusion_matrix(cm)
    # print(len(total_features))

    return np.array(total_features)



