from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.colors import ListedColormap
import csv

from app import init_pre_processing, init_activity_detection, init_feature_extraction, init_classificator, main
from dataset import get_dataset, dataset_analyzer
from interfaces import feature_extraction

import pickle
# ML
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
from utils import boxplot, read_csv, plot_odf

mfcc = 10

log_file = 'evaluation_logs/classificator_evaluation/model_all_features.csv'
model_normalization = './app/model_normalization_all_features.csv'


def classify_and_plot(data):
    with open(log_file, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['audio_class', 'duration', 'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
                  'mfcc_mean_6',
                  'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3',
                  'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10',
                  'rms_mean', 'rms_std', 'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
                  'spectral_rolloff_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                  'spectral_contrast_mean_1', 'spectral_contrast_mean_2', 'spectral_contrast_mean_3',
                  'spectral_contrast_mean_4', 'spectral_contrast_mean_5', 'spectral_contrast_mean_6',
                  'spectral_contrast_mean_7', 'spectral_contrast_std_1', 'spectral_contrast_std_2',
                  'spectral_contrast_std_3', 'spectral_contrast_std_4', 'spectral_contrast_std_5',
                  'spectral_contrast_std_6', 'spectral_contrast_std_7', 'spectral_flatness_mean',
                  'spectral_flatness_std']
        writer.writerow(header)
        # close the file
        f.close()

    with open(model_normalization, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        header = ['duration', 'mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'mfcc_mean_4', 'mfcc_mean_5',
                  'mfcc_mean_6',
                  'mfcc_mean_7', 'mfcc_mean_8', 'mfcc_mean_9', 'mfcc_mean_10', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_std_3',
                  'mfcc_std_4', 'mfcc_std_5', 'mfcc_std_6', 'mfcc_std_7', 'mfcc_std_8', 'mfcc_std_9', 'mfcc_std_10',
                  'rms_mean', 'rms_std', 'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
                  'spectral_rolloff_std', 'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                  'spectral_contrast_mean_1', 'spectral_contrast_mean_2', 'spectral_contrast_mean_3',
                  'spectral_contrast_mean_4', 'spectral_contrast_mean_5', 'spectral_contrast_mean_6',
                  'spectral_contrast_mean_7', 'spectral_contrast_std_1', 'spectral_contrast_std_2',
                  'spectral_contrast_std_3', 'spectral_contrast_std_4', 'spectral_contrast_std_5',
                  'spectral_contrast_std_6', 'spectral_contrast_std_7', 'spectral_flatness_mean',
                  'spectral_flatness_std']
        writer.writerow(header)
        f.close()

    X = []
    Y = []

    for audio in data:

        # Init system
        # init_pre_processing()
        # init_activity_detection()
        # init_feature_extraction(n_mfcc_arg=mfcc)
        # init_classificator(by_pass=True)
        # buffer_len = 512
        #
        # # Call system
        # response = main(audio, buffer_len)

        features = feature_extraction("all", audio.waveform, audio.sample_rate, 10, 512, [])

        # features = response['FEATURES']

        if len(features) == 45:
            # plot_audio(audio.waveform,response['SIGNAL_PROCESSED'],audio.sample_rate)
            # plot_odf(audio.filename,audio.waveform,response['SIGNAL_PROCESSED'],audio.sample_rate,response['ONSET_LOCATIONS'],response['HFC'],response['THRESHOLD'])
            row = [audio.class_type]
            row.extend(features)

            # Store features values for evaluation
            with open(log_file, 'a+', newline='') as file:
                w = csv.writer(file)
                w.writerow(row)
                file.close()

            X.append(features)
            Y.append(audio.class_type)
        else:
            print(len(features))

    X = np.array(normalize(X))
    Y = np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # init vars
    n_neighbors = 3
    h = .1  # step size in the mesh

    sss = X[:, 0]

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    cmap_bold = ['darkorange', 'c', 'darkblue']

    rcParams['figure.figsize'] = 5, 5
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        knn_classifier = KNeighborsClassifier(n_neighbors, weights=weights)
        knn_classifier.fit(X, Y)

        # evaluate
        y_expected = y_test
        y_predicted = knn_classifier.predict(X_test)

        # print results
        print('----------------------------------------------------------------------')
        print('Classification report')
        print('----------------------------------------------------------------------')
        print('\n', classification_report(y_expected, y_predicted))
        print('----------------------------------------------------------------------')

    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()

    return knn_classifier


def features_boxplot():
    df = pd.read_csv(log_file)
    kick = df[df['audio_class'] == 'Kick']
    hh = df[df['audio_class'] == 'HH']
    snare = df[df['audio_class'] == 'Snare']
    for column in df:
        if column != 'audio_class':
            data = [kick[column].T, hh[column].T, snare[column].T]
            boxplot(data, column)


def save_normalization(array):
    # Store features values for evaluation
    with open(model_normalization, 'a+', newline='') as file:
        w = csv.writer(file)
        w.writerow(np.max(array, axis=0))
        w.writerow(np.min(array, axis=0))
        file.close()


def normalize(data):
    data_norm = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    save_normalization(data)
    return data_norm


# dataset_analyzer(data)
knn_model = classify_and_plot(get_dataset())
# features_boxplot()
# save the model to disk
filename = './app/finalized_model_all_features.sav'
pickle.dump(knn_model, open(filename, 'wb'))
