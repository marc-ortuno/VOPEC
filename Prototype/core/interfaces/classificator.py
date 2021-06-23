import numpy as np


def classificator(features, model):
    """
    Classificator interface
    :param features: Array of features
    :param model : KNeighborsClassifier
    :output class_tag: Int
    """
    features = np.array(features)
    features = features.reshape(1, features.size)  # (1,20)
    class_tag = model.predict(features)
    return class_tag[0]
