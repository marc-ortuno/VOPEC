import numpy as np
from dataset import get_dataset, dataset_analyzer
from interfaces import pre_processing, activity_detection, feature_extraction, classificator

from sklearn.metrics import classification_report


def init_pre_processing(by_pass = False):
    global pre_processing_by_pass
    pre_processing_by_pass = by_pass

def init_activity_detection(func_type = 1, by_pass = False):

    #Activity detection variables
    global onset_location
    global last_onset
    global hfc
    global activity_detection_type #Function name
    global activiy_detection_by_pass
    global previous_hfc
    global previous_th


    onset_location = []
    last_onset = False
    hfc = 0
    previous_hfc = []
    previous_th = []
    activity_detection_type = func_type
    activiy_detection_by_pass = by_pass

def init_feature_extraction(n_mfcc_arg = 20, by_pass = False):
    #Feature extraction variables
    global active_signal
    global features
    global n_mfcc
    global feature_extraction_by_pass

    active_signal = []
    features = []
    n_mfcc = n_mfcc_arg
    feature_extraction_by_pass = by_pass

def init_classificator(knn_model = [], by_pass = False):
    #Classificator Variables
    global model
    global predicted
    global classificator_by_pass

    model = knn_model
    predicted = []
    classificator_by_pass = by_pass

# Init audio process variables
def init(sr,b_len):

    # declare variables used in `process`
    #Audio details
    global samp_freq
    global buffer_len
    global onset_timeout
    global onset_duration

    samp_freq = sr
    buffer_len = b_len
    avg_duration = 0.190 #in seconds
    onset_duration = int(avg_duration / (b_len / sr)) #150ms average duration of a class
    onset_timeout = onset_duration


# the process function!
def process(input_buffer, output_buffer):

    global last_onset
    global active_signal
    global onset_timeout

    features = []
    activity_detected = False
    
    if not pre_processing_by_pass:

        #Pre-Processing Block
        n_signal = pre_processing(input_buffer,samp_freq)

        if not activiy_detection_by_pass:
            
            #Activity Detection Block
            onset, hfc, threshold = activity_detection(activity_detection_type,n_signal,samp_freq,buffer_len,previous_hfc)
            previous_hfc.append(hfc)
            previous_th.append(threshold)
            # To prevent repeated reporting of an
            # onset (and thus producing numerous false positive detections), an
            # onset is only reported if no onsets have been detected in the previous three frames (30 ms aprox).
            th = previous_th[-2:]

            if last_onset is True and onset is False:
                if onset_timeout > 0:
                    onset = True
                    onset_timeout -= 1
                    activity_detected = False
                else:
                    onset_timeout = onset_duration
                    activity_detected = True

                if len(th) > 1 and int(th[1]) < int(th[0]):
                    onset = False
                    onset_timeout = onset_duration
                    activity_detected = True
                    



            if onset:
                active_signal.extend(input_buffer) # Until we get an offset, the active sound is stored for later do feature extraction an onset is False: classification.
                onset_location.extend(np.ones(buffer_len)) # Onset location for visual analysis    
            else:
                onset_location.extend(np.zeros(buffer_len))

            #Offset detected
            if (activity_detected):
                
                print("ACTIVITY DETECTED")
                if not feature_extraction_by_pass:
                    
                    #Feature Extraction Block
                    features = feature_extraction(active_signal,samp_freq,n_mfcc)
                    active_signal = [] #Clean active signal buffer
                    
                    if not classificator_by_pass:
                        
                        #Classificator Block
                         predicted.append(classificator(features,model))
            
            last_onset = onset


    return n_signal, features, hfc, predicted, onset_location, threshold
    
def main(audio, buffer_len = 512):
    
    #Signal details
    signal = audio.waveform
    samp_freq = audio.sample_rate
    n_buffers = len(signal)//buffer_len

    #Init process variables
    init(samp_freq, buffer_len)
    
    data_type = signal.dtype
    # allocate input and output buffers
    input_buffer = np.zeros(buffer_len, dtype=data_type)
    output_buffer = np.zeros(buffer_len, dtype=data_type)
    
    onset_location = []
    total_features = []
    total_hfc = []
    total_th = []
    # simulate block based processing
    signal_proc = np.zeros(n_buffers*buffer_len, dtype=data_type)
    
    for k in range(n_buffers):

        # index the appropriate samples
        input_buffer = signal[k*buffer_len:(k+1)*buffer_len]
        output_buffer, features, hfc, predicted, onset_location, threshold = process(input_buffer, output_buffer)

        signal_proc[k*buffer_len:(k+1)*buffer_len] = output_buffer

        total_features.extend(features)

        if type(hfc) is np.ndarray:
            total_hfc.extend(hfc)
        else:
            total_hfc.extend([hfc]*output_buffer.size)

        if type(threshold) is np.ndarray:
            total_th.extend(threshold)
        else:
            total_th.extend([threshold]*output_buffer.size)

    #return in a dictionary
    return {'SIGNAL_PROCESSED':signal_proc, 'ONSET_LOCATIONS':onset_location, 'HFC':total_hfc, 'THRESHOLD':total_th, 'PREDICTION':predicted, 'FEATURES':features}



