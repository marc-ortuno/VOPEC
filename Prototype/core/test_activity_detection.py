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
tests_dir = './test_logs'

#create root folder
if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)

def run_test(wav_dir,csv_dir,buffer_size,log_file):
    '''
    Run test which generates a csv with the results
    '''

    #Load audio and its annotation
    audio = Waveform(path=wav_dir)
    groundtruth_annotation = load_annotation(csv_dir)

    #run simulation
    init_pre_processing()
    init_activity_detection(func_type=2)
    init_feature_extraction(by_pass=True)
    init_classificator(by_pass=True)
    result = main(audio,buffer_size)

    #Init groundtruth activity array
    groundtruth_activity = np.zeros(len(result['ONSET_LOCATIONS']))
    sample_rate = audio.sample_rate

    for i in range(0,len(groundtruth_annotation),2):
        sample_instant_1 = int(float(groundtruth_annotation[i][0])*sample_rate)
        sample_instant_2 = int(float(groundtruth_annotation[i+1][0])*sample_rate)
        groundtruth_activity[sample_instant_1:sample_instant_2] = 1

    #evaluate activity detection
    precision,recall,f1_score,accuracy= evaluate_activity_detection(audio.waveform,groundtruth_activity,result['ONSET_LOCATIONS'],sample_rate)

    row = [wav_dir, precision,recall,f1_score,accuracy]

    with open(log_file, 'a+',newline='') as file:
        w = csv.writer(file)
        w.writerow(row)
        file.close()

def all_dataset_test(buffer_size = 512):
    startpath = "../../RawDataset"
    #Create dataset_log.csv file where all the metadata will be located.
    log_file = tests_dir+'/activity_detection_log_'+str(buffer_size)+'.csv'

    with open(log_file, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['Audio', 'Precision', 'Recall', 'F1-Score','Accuracy']
        writer.writerow(header)
        # close the file
        f.close()

    for root, _, files in os.walk(startpath):
            folder = '/' + os.path.basename(root) + '/'
            # #TODO: Optimize parsing csv and its wav (currently double for...)
            for f in files:
                if f.endswith('.wav') and os.path.isfile(startpath+folder+f.split('.')[0]+'.csv'):
                    wav_dir = startpath+folder+f
                    print(wav_dir)
                    csv_dir = startpath+folder+f.split('.')[0]+'.csv'
                    run_test(wav_dir,csv_dir,buffer_size,log_file)

def generate_plots():
    evaluation_csv = read_csv(tests_dir+'/activity_detection_log_512.csv')
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


def buffer_size_test():

    buffer_sizes = [128, 256, 512, 1024, 2048, 4096]

    for buffer_size in buffer_sizes:
        all_dataset_test(buffer_size=buffer_size)



# buffer_size_test()
all_dataset_test()
generate_plots()

# run_test('../../RawDataset/SOF/SofI2.wav,','../../RawDataset/SOF/SofI2.wav,',512)