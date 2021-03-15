from matplotlib import pyplot as plt
from interfaces import feature_extraction
import numpy as np
#ML
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def train_model(data):
    X = []
    Y = []
    for sound in data:
        X.append(feature_extraction(sound.waveform,sound.sample_rate,40))
        Y.append(sound.class_type)

    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X,Y)


    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(x_train, y_train)

    y_pred = knn_classifier.predict(x_test)
    print(classification_report(y_pred,y_test))

    return knn_classifier