from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from sklearn.metrics import f1_score


def evaluate_system(groundtruth,predicted):
    """
    Tests to prove the system
    Precision
    Recall
    F-Score
    """
    f1_score(groundtruth, predicted, average='macro')
