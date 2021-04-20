from matplotlib import pyplot as plt
import numpy as np
from dataset import get_dataset, dataset_analyzer
from app import main, init
import pickle
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
        features = (main(sound,[]))
        X.append(features)
        Y.append(sound.class_type)

    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X,Y)
    
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(x_train, y_train)

    y_pred = knn_classifier.predict(x_test)
    print(classification_report(y_pred,y_test))

    return knn_classifier



#Pre train k-NN
data = get_dataset()
dataset_analyzer(data)
knn_model = train_model(data)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(knn_model, open(filename, 'wb'))
