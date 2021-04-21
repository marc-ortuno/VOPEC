from evaluate import evaluate_activity_detection
from app import main
from models import Waveform
from utils import load_groundtruth,load_annotation
import numpy as np 
import os
import csv

'''
This script
'''

#save test metrics in a csv to then make
tests_dir = '../test_logs'

#create root folder
if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)

#Create dataset_log.csv file where all the metadata will be located.
with open(tests_dir+'/activity_detection_log.csv', 'w', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    header = ['Audio', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
    writer.writerow(header)
    # close the file
    f.close()

startpath = "../../RawDataset"

for root, dirs, files in os.walk(startpath):
        folder = '/' + os.path.basename(root) + '/'
        # #TODO: Optimize parsing csv and its wav (currently double for...)
        for f in files:
            if f.endswith('.wav') and os.path.isfile(startpath+folder+f.split('.')[0]+'.csv'):
                wav_dir = startpath+folder+f
                csv_dir = startpath+folder+f.split('.')[0]+'.csv'

                #Load audio and its annotation
                audio = Waveform(path=wav_dir)
                groundtruth = load_groundtruth(csv_dir)
                groundtruth_annotation = load_annotation(csv_dir)

                #run simulation
                predicted_activity = main(audio,groundtruth)

                #Init groundtruth activity array
                groundtruth_activity = np.zeros(len(predicted_activity))
                sample_rate = audio.sample_rate

                for i in range(0,len(groundtruth_annotation),2):
                    sample_instant_1 = int(float(groundtruth_annotation[i][0])*sample_rate)
                    sample_instant_2 = int(float(groundtruth_annotation[i+1][0])*sample_rate)
                    groundtruth_activity[sample_instant_1:sample_instant_2] = 1

                #evaluate activity detection
                precision,recall,f1_score,accuracy = evaluate_activity_detection(audio.waveform,groundtruth_activity,predicted_activity,sample_rate)

                row = [wav_dir, precision,recall,f1_score,accuracy]

                with open(tests_dir+'/activity_detection_log.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)







