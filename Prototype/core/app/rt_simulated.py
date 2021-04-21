import numpy as np
from dataset import get_dataset, dataset_analyzer
from interfaces import pre_processing, activity_detection, feature_extraction, classificator

from sklearn.metrics import classification_report


# state variables
def init(signal,sr,n_buffers, knn_model, b_len,pp_by_pass, ad_by_pass,fe_by_pass,c_by_pass):

    # declare variables used in `process`
    #Audio details
    global samp_freq
    global buffer_len
    global audio_size

    samp_freq = sr
    buffer_len = b_len
    audio_size = len(signal)


    #Activity detection variables
    global onset_location
    global n_fft
    global hop_size
    global last_onset
    global hfc

    onset_location = []
    n_fft = 256
    hop_size = 64
    last_onset = False
    hfc = 0

    #Feature extraction variables
    global active_signal
    global features
    global n_mfcc

    active_signal = []
    features = []
    n_mfcc = 20

    #Classificator Variables
    global model
    global predicted

    model = knn_model
    predicted = []

    #Bypass
    global pre_processing_by_pass
    global activiy_detection_by_pass
    global feature_extraction_by_pass
    global classificator_by_pass
    pre_processing_by_pass = pp_by_pass
    activiy_detection_by_pass = ad_by_pass
    feature_extraction_by_pass = fe_by_pass
    classificator_by_pass = c_by_pass


# the process function!
def process(input_buffer, output_buffer):

    global last_onset
    global active_signal

    features = []

    if not pre_processing_by_pass:

        #Pre-Processing Block
        n_signal = pre_processing(input_buffer,samp_freq)

        if not activiy_detection_by_pass:
            
            #Activity Detection Block
            onset, hfc = activity_detection(n_signal,samp_freq,n_fft,hop_size)

            # Update buffer containing the active signal while is an onset
            if onset:
                active_signal.extend(input_buffer) # Until we get an offset, the active sound is stored for later do feature extraction and classification.
                onset_location.extend(np.ones(buffer_len)) # Onset location for visual analysis
            else:
                onset_location.extend(np.zeros(buffer_len))
            
            #Offset detected
            if (last_onset is True and onset is False) or (len(active_signal) >= audio_size):
                
                if not feature_extraction_by_pass:
                    
                    #Feature Extraction Block
                    features = feature_extraction(active_signal,samp_freq,n_mfcc)
                    active_signal = [] #Clean active signal buffer
                    
                    if not classificator_by_pass:
                        
                        #Classificator Block
                         predicted.append(classificator(features,model))
            
            last_onset = onset


    return n_signal, features, hfc, predicted, onset_location
    
def main(signal,samp_freq,n_buffers, buffer_len = 512, knn_model = [],
            pre_processing_by_pass = False, activiy_detection_by_pass = False ,feature_extraction_by_pass = False,classificator_by_pass = False):
    
    #Init process variables
    init(signal,samp_freq,n_buffers, knn_model, buffer_len,pre_processing_by_pass, activiy_detection_by_pass,feature_extraction_by_pass,classificator_by_pass)
    
    data_type = signal.dtype
    # allocate input and output buffers
    input_buffer = np.zeros(buffer_len, dtype=data_type)
    output_buffer = np.zeros(buffer_len, dtype=data_type)
    
    onset_location = []
    total_features = []
    total_hfc = []
    # simulate block based processing
    signal_proc = np.zeros(n_buffers*buffer_len, dtype=data_type)
    
    for k in range(n_buffers):

        # index the appropriate samples
        input_buffer = signal[k*buffer_len:(k+1)*buffer_len]
        output_buffer, features, hfc, predicted, onset_location = process(input_buffer, output_buffer)
        signal_proc[k*buffer_len:(k+1)*buffer_len] = output_buffer
        total_features.extend(features)
        total_hfc.extend([hfc]*output_buffer.size)

    return signal_proc, onset_location, total_hfc, predicted, features



