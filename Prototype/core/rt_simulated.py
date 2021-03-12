import numpy as np
from models import Waveform
from dataset import get_dataset
from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from utils import plot_audio, plot_spectrum, plot_odf,plot_librosa_spectrum,plot_fft

# parameters
buffer_len = 512

# test signal
audio = Waveform(path="./data/AFRP2.wav")
signal = audio.waveform
samp_freq = audio.sample_rate
duration = audio.duration

n_buffers = len(signal)//buffer_len
data_type = signal.dtype
# allocate input and output buffers
input_buffer = np.zeros(buffer_len, dtype=data_type)
output_buffer = np.zeros(buffer_len, dtype=data_type)
onset_location = []



# state variables
def init():

    # declare variables used in `process`
    # global
    global data
    global groundtruth
    global predicted 
    global n_fft
    global hop_size
    global features
    global last_onset 
    global hfc

    data = get_dataset(sound="Kick",microphone="2") 
    groundtruth = []
    predicted = []

    n_fft = 256
    hop_size = 64
    features = []
    last_onset = False
    hfc = []

    return


# the process function!
def process(input_buffer, output_buffer, buffer_len):

    global last_onset
    global hfc
    global features
    global n_fft
    global hop_size

    n_signal = pre_processing(input_buffer,samp_freq)

    #Detect onset
    onset = activity_detection(n_signal,samp_freq,n_fft,hop_size)

    # Only calculated when the onset is detected
    if onset:
        # print("onset")
        onset_location.extend(np.ones(buffer_len))
        w_features = feature_extraction(n_signal)
        features.append(w_features)
    else:
        onset_location.extend(np.zeros(buffer_len))

    #Offset detected
    if last_onset is True and onset is False:
        predicted.append(classificator(features))
    
    last_onset = onset

    output_buffer = n_signal
    return output_buffer



init()
# simulate block based processing
signal_proc = np.zeros(n_buffers*buffer_len, dtype=data_type)
for k in range(n_buffers):

    # index the appropriate samples
    input_buffer = signal[k*buffer_len:(k+1)*buffer_len]
    output_buffer = process(input_buffer, output_buffer, buffer_len)
    signal_proc[k*buffer_len:(k+1)*buffer_len] = output_buffer

plot_audio(signal,signal_proc,samp_freq)
plot_odf(signal_proc,samp_freq,onset_location)