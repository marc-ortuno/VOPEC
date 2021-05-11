from matplotlib import pyplot as plt
import numpy as np
from dataset import get_dataset, dataset_analyzer
from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from utils import plot_audio, plot_spectrum, plot_odf, plot_librosa_spectrum, plot_fft, plot_confusion_matrix, \
    plot_evaluation_report
import pickle
# ML
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

mfcc = 20


def train_model(data):
    X = []
    Y = []

    for audio in data:

        # Init system
        init_pre_processing()
        init_activity_detection(func_type=1)
        init_feature_extraction(n_mfcc_arg=mfcc)
        init_classificator(by_pass=True)
        buffer_len = 512

        # Call system
        response = main(audio, buffer_len)

        features = response['FEATURES']
        if len(features) == 42:
            # plot_audio(audio.waveform,response['SIGNAL_PROCESSED'],audio.sample_rate)
            X.append(features)
            Y.append(audio.class_type)

    # X = np.array(X)
    # Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    knn_classifier = KNeighborsClassifier(weights="distance")
    knn_classifier.fit(x_train, y_train)

    y_pred = knn_classifier.predict(x_test)
    print(classification_report(y_pred, y_test))

    return knn_classifier


# Pre train k-NN
data = get_dataset()
# dataset_analyzer(data)
knn_model = train_model(data)

# save the model to disk
filename = './app/finalized_model_v2.sav'
pickle.dump(knn_model, open(filename, 'wb'))
