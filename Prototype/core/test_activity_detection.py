from evaluate import evaluate_activity_detection
from app import main,init_activity_detection,init_classificator,init_feature_extraction,init_pre_processing
from models import Waveform
from utils import load_groundtruth,load_annotation,read_csv, plot_boxplot
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

def run_test():
    '''
    Run test which generates a csv with the results
    '''
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
                    groundtruth_annotation = load_annotation(csv_dir)

                    #Signal details
                    signal = audio.waveform
                    samp_freq = audio.sample_rate
                    buffer_len = 512
                    n_buffers = len(signal)//buffer_len

                    #run simulation
                    init_pre_processing()
                    init_activity_detection()
                    init_feature_extraction()
                    init_classificator(by_pass=True)
                    predicted_activity= main(signal,samp_freq,n_buffers,buffer_len)[1]

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


def generate_plots():
    evaluation_csv = read_csv(tests_dir+'/activity_detection_log.csv')
    precision = []
    recall = []
    f1_score = []
    accuracy = []

    for i in range(1,len(evaluation_csv),1):
        precision.append(evaluation_csv[i][1])
        recall.append(evaluation_csv[i][2])
        f1_score.append(evaluation_csv[i][3])
        accuracy.append(evaluation_csv[i][4])


    plot_boxplot(precision,recall,f1_score,accuracy)



run_test()
#generate_plots()