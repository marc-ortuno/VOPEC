import numpy as np
from utils import plot_dataset_statistics
import librosa.feature as lf

def dataset_analyzer(dataset):
    #Init variables
    class_types = ['Kick','HH','Snare']
    class_type_len = len(class_types)

    n_sounds = len(dataset)
    total_duration = 0
    av_duration = 0
    n_sounds_class = list(range(class_type_len))
    total_duration_class = list(range(class_type_len))
    av_duration_class = list(range(class_type_len))

    for sound in dataset:
        for class_index in range(class_type_len):
            if sound.class_type == class_types[class_index]:
                n_sounds_class[class_index]+=1
                total_duration_class[class_index]+=sound.duration
                
    total_duration = sum(total_duration_class)
    av_duration = total_duration / n_sounds
    av_duration_class = [x/y for x, y in zip(total_duration_class, n_sounds_class)]
    

    #Only for plot 
    n_sounds_class.append(n_sounds)
    total_duration_class.append(total_duration)
    av_duration_class.append(av_duration)

    data = [n_sounds_class,total_duration_class,av_duration_class]
    plot_dataset_statistics(class_types,data)