import matplotlib.pyplot as plt

from evaluate import evaluate_activity_detection, evaluate_system
from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from models import Waveform
from utils import load_groundtruth, load_annotation, read_csv, plot_metrics_boxplot, plot_confusion_matrix, \
    plot_metrics_classification_boxplot
import numpy as np
import os
import csv
import pickle
import pandas as pd

'''
Classificator/system evaluation

The purpose of this script is to obtain the precision, recall, f-score and accuracy metrics of the beatbox 
classification interface by comparing the classificator output with the corresponding groundtruth (annotation file).

'''

# save test metrics in a csv to then make
tests_dir = 'evaluation_logs/classificator_evaluation'
# Import model
filename = './app/finalized_model_all_features.sav'
knn_model = pickle.load(open(filename, 'rb'))

model_normalization = './app/model_normalization_all_features.csv'
normalization_values = pd.read_csv(model_normalization)
# create root folder
if not os.path.exists(tests_dir):
    os.makedirs(tests_dir)


def run_test(wav_dir, csv_dir, buffer_size, log_file):
    '''
    Run test function:
    input:
        - wav_dir: Location of the audio
        - csv_dir: Location of the csv annotation
        - buffer_size: Default is 512 but it can be modified to test the system on different buffer sizes
        - log_file: Location of the file where all results are logged.
    '''

    # Load audio and its annotation
    audio = Waveform(path=wav_dir)
    groundtruth = read_csv(csv_dir)

    # Init system simulation
    init_pre_processing()
    init_activity_detection()
    init_feature_extraction(func_type="all", n_mfcc_arg=10, norm_file=normalization_values)
    init_classificator(knn_model=knn_model)

    # run simulation
    result = main(audio, buffer_size)

    prediction = get_prediction_time_instants(result['ONSET_LOCATIONS'], result['PREDICTION'], audio.sample_rate)

    # evaluate activity detection
    precision, recall, fscore, cm, fm = evaluate_system(groundtruth, prediction)
    #plot_confusion_matrix(cm)

    row = [wav_dir, precision, recall, fscore]

    with open(log_file, 'a+', newline='') as file:
        w = csv.writer(file)
        w.writerow(row)
        file.close()


def all_dataset_test(startpath, buffer_size=512, proposal="mfcc"):
    '''
    all_dataset_test:
    input:
        - startpath: root directory of audios
        - buffer_size: test the system on different buffer sizes

    given a directory run test for each audio, results are stored in the log file
    '''

    # Create dataset_log.csv file where all the metadata will be located.
    log_file = tests_dir + '/classification_log_' + str(proposal) + '.csv'

    with open(log_file, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['Audio', 'Precision', 'Recall', 'F1-Score']
        writer.writerow(header)
        # close the file
        f.close()

    for root, _, files in os.walk(startpath):
        folder = '/' + os.path.basename(root) + '/'
        # #TODO: Optimize parsing csv and its wav (currently double for...)
        for f in files:
            if f.endswith('.wav') and os.path.isfile(startpath + folder + f.split('.')[0] + '.csv'):
                wav_dir = startpath + folder + f
                csv_dir = startpath + folder + f.split('.')[0] + '.csv'
                run_test(wav_dir, csv_dir, buffer_size, log_file)


def generate_plots():
    '''
    Read log file and creates a boxplot
    '''
    proposals = ["mfcc", "all"]
    for proposal in proposals:
        evaluation_csv = read_csv(tests_dir + '/classification_log_' + str(proposal) + '.csv')
        precision = []
        recall = []
        f1_score = []
        accuracy = []

        for i in range(1, len(evaluation_csv), 1):
            precision.append(evaluation_csv[i][1])
            recall.append(evaluation_csv[i][2])
            f1_score.append(evaluation_csv[i][3])

        plot_metrics_classification_boxplot(precision, recall, f1_score, proposal)


def get_prediction_time_instants(onset_location, prediction, sr):
    time_list = []
    last_value = 0
    prediction_list = []
    current_index = 0
    for i in range(0, len(onset_location)):
        if onset_location[i] == 1 and last_value == 0:
            time_list.append(i / sr)
        elif onset_location[i] == 0 and last_value == 1:
            time_list.append(i / sr)
        last_value = onset_location[i]

    for j in range(0, len(prediction)):
        prediction_list.append([time_list[j+current_index], prediction[j]])
        prediction_list.append([time_list[j+current_index+1], prediction[j]])
        current_index += 1

    return prediction_list


startpath = "../../RawDataset"  # Root dir of test audios

# Run tests
#all_dataset_test(startpath, proposal="all")

# Save plots
generate_plots()
