from evaluate import evaluate_activity_detection
from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from models import Waveform
from utils import load_groundtruth, load_annotation, read_csv, plot_metrics_boxplot
import numpy as np
import os
import csv

'''
Activity detection evaluation

The purpose of this script is to obtain the precision, recall, f-score and accuracy metrics of the beatbox activity detection interface
by comparing the activity detection output with the corresponding groundtruth (annotation file).

To obtain the activity detection output, we run the system simulation through an audio or multiple audios.

An initialization of the system makes possible to bypass non-desired interfaces of the system, as the feature extraction or the classification stages.

The main script (the simulation script) is designed in such a way that returns an array of the detected activity  (1 activity, 0 non-activity), we obtain this array and then
compare it with the grountruth annotation which is provided by the csv annotation of the audio.
'''

# save test metrics in a csv to then make
tests_dir = 'evaluation_logs/activity_detection_evaluation'

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
    print(wav_dir)
    audio = Waveform(path=wav_dir)
    groundtruth_annotation = load_annotation(csv_dir)

    # Init system simulation
    init_pre_processing()
    init_activity_detection(func_type=1)
    init_feature_extraction(by_pass=True)
    init_classificator(by_pass=True)

    # run simulation
    result = main(audio, buffer_size)

    # Init groundtruth activity array
    groundtruth_activity = np.zeros(len(result['ONSET_LOCATIONS']))
    sample_rate = audio.sample_rate

    # Transform annotation in the desired format (1 activity, 0 non-activity)
    for i in range(0, len(groundtruth_annotation), 2):
        sample_instant_1 = int(float(groundtruth_annotation[i][0]) * sample_rate)
        sample_instant_2 = int(float(groundtruth_annotation[i + 1][0]) * sample_rate)
        groundtruth_activity[sample_instant_1:sample_instant_2] = 1

    # evaluate activity detection
    precision, recall, f1_score, accuracy = evaluate_activity_detection(groundtruth_activity, result['ONSET_LOCATIONS'])

    row = [wav_dir, precision, recall, f1_score, accuracy]

    with open(log_file, 'a+', newline='') as file:
        w = csv.writer(file)
        w.writerow(row)
        file.close()


def all_dataset_test(startpath, buffer_size=512, proposal=3):
    '''
    all_dataset_test:
    input:
        - startpath: root directory of audios
        - buffer_size: test the system on different buffer sizes

    given a directory run test for each audio, results are stored in the log file
    '''

    # Create dataset_log.csv file where all the metadata will be located.
    log_file = tests_dir + '/proposal_' + str(proposal) + '/activity_detection_log_' + str(buffer_size) + '.csv'

    with open(log_file, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['Audio', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
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


def generate_plots(buffer_sizes, proposal=3):
    '''
    Read log file and creates a boxplot
    '''
    for buffer_size in buffer_sizes:
        evaluation_csv = read_csv(
            tests_dir + '/proposal_' + str(proposal) + '/activity_detection_log_' + str(buffer_size) + '.csv')
        precision = []
        recall = []
        f1_score = []
        accuracy = []

        for i in range(1, len(evaluation_csv), 1):
            precision.append(evaluation_csv[i][1])
            recall.append(evaluation_csv[i][2])
            f1_score.append(evaluation_csv[i][3])
            accuracy.append(evaluation_csv[i][4])

        plot_metrics_boxplot(precision, recall, f1_score, accuracy, buffer_size)


def buffer_size_test(path, buffer_sizes):
    # Run all dataset_test with different buffer size

    for buffer_size in buffer_sizes:
        all_dataset_test(path, buffer_size=buffer_size, proposal=3)


startpath = "../../RawDataset"  # Root dir of test audios
buffer_sizes = [512]  # Different buffer size of the test

buffer_size_test(startpath, buffer_sizes)

# Run tests
#all_dataset_test(startpath)

# Save plots
generate_plots(buffer_sizes, proposal=3)
