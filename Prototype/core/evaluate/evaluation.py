from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from sklearn.metrics import f1_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from utils import plot_ad_evaluation,plot_evaluation_report
import sys

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
        sys.exit("Groundtruth and predicted have different sizes. Check activity detection!")
    
    return report, cm

def evaluate_activity_detection(signal,groundtruth,predicted,sr):
    '''
    Evaluate activity detection
    
    '''
    
    #Metrics
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for index in range(0,len(groundtruth),1):
        if groundtruth[index] == predicted[index] and predicted[index] == 1:
            TP += 1
        elif groundtruth[index] == predicted[index] and predicted[index] == 0:
            TN += 1
        elif groundtruth[index] != predicted[index] and predicted[index] == 0:
            FN += 1
        elif groundtruth[index] != predicted[index] and predicted[index] == 1:
            FP += 1 

    if (TP == 0 and FP == 0) or (TP ==0 and FN == 0):
        precision = 0
        recall = 0
        f1_score = 0
        accuracy = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        accuracy = (TP + TN)/(TP + TN + FP + FN)
              

    # print("Precision " + str(precision))
    # print("Recall " + str(recall))
    # print("F1_score " + str(f1_score))
    # print("Accuracy " +str(accuracy))

    return precision,recall,f1_score,accuracy
    #plot_ad_evaluation(signal,groundtruth,predicted,sr)
