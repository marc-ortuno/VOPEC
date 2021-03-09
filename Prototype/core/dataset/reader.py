import os
import wave
from scipy.io import wavfile
from models import Waveform

""" 
Returns an array of sounds directories. 
The database can be filtered by sound and/or microphone
"""
def read_dataset(self,sound = None ,microphone = None):
    dataset_path = "../../Dataset/"
    dir_output = []
    for root, dirs, files in os.walk(dataset_path):
        folder = os.path.basename(root)
        if sound is not None and microphone is not None:
            if sound == folder:
                for f in files:
                    file_name = f.split('.')[0]
                    if file_name[-1] is microphone:
                        dir_output.append(folder+"/"+f)
        elif sound is not None and microphone is None:
            if sound == folder:
                for f in files:
                    dir_output.append(folder+"/"+f)
        elif sound is None and microphone is not None:
            for f in files:
                    file_name = f.split('.')[0]
                    if file_name[-1] is microphone:
                        dir_output.append(folder+"/"+f)
        else:
            for f in files:
                dir_output.append(folder+"/"+f)      
    return dir_output

"""
Returns an array of Audio Objects.
Each Audio object contains a wav audio, the rate, filename and url of the file.
"""
def get_dataset(self,sound = None ,microphone = None):
    root_path = "../../Dataset/"
    dir_dataset = read_dataset(sound,microphone)
    dataset = []
    for file in dir_dataset:
        audio = Waveform(path=root_path+file,filename=file.split('/')[1])
        dataset.append(audio)  
    return dataset

