from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

def evaluate_system(groundtruth,predicted):
    """
    Tests to prove the system
    Precision
    Recall
    F-Score
    """
    if len(predicted) == len(groundtruth):
        report = classification_report(groundtruth,predicted,output_dict=True)
        cm = confusion_matrix(groundtruth,predicted)

    else:
        print("Groundtruth and predicted have different sizes")
    
    return report, cm

