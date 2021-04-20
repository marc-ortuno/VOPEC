import os
import wave
from scipy.io import wavfile
from models import Waveform

""" 
Returns an array of sounds directories. 
The database can be filtered by sound and/or microphone
"""
def read_dataset(sound = None ,microphone = None,root_path= None) :
    dataset_path = root_path
    dir_output = []
    for root, dirs, files in os.walk(dataset_path):
        folder = os.path.basename(root)
        if folder != 'Metadata':
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
Returns an array of Waveform Objects (surfboard library).
"""
def get_dataset(sound = None ,microphone = None):
    root_path = "../../Dataset/"
    dir_dataset = read_dataset(sound,microphone,root_path)
    dataset = []
    for file in dir_dataset:
        class_type =file.split('/')[0]
        filename=file.split('/')[1]
        audio = Waveform(path=root_path+file,filename=filename,class_type = class_type)
        dataset.append(audio)  
    return dataset

