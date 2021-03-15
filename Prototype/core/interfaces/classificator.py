import numpy as np


def classificator(features,model):
    """
    Pre-processing interface
    :param features: Array of features
    :param model : KNeighborsClassifier
    :output class_tag: Int
    """
    features = features.reshape(1, features.size) #(1,40)
    class_tag = model.predict(features)
    return class_tag[0]   