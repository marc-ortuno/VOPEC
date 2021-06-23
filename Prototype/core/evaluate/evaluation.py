from interfaces import pre_processing, activity_detection, feature_extraction, classificator
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from utils import plot_ad_evaluation, plot_evaluation_report
import sys
import numpy as np


def event_assignment(groundtruth, predicted):
    """
    This algorithm solves the problem of assigning a prediction class with its corresponding groundtruth.
    """
    matrix = np.zeros((len(predicted) // 2, len(groundtruth) // 2))

    # Matriz de coste (similarity)
    for i in range(0, len(predicted), 2):
        ii = i // 2
        predicted_onset = predicted[i][0]
        for j in range(0, len(groundtruth), 2):
            jj = j // 2
            groundtruth_onset = float(groundtruth[j][0])
            distance = np.abs(groundtruth_onset - predicted_onset)
            matrix[ii, jj] = distance

    # plt.imshow(matrix)
    # plt.colorbar()
    # plt.ylabel("Predicted")
    # plt.xlabel("Ground truth")
    # plt.show()

    th = 0.05
    assignment_index = []
    for row in range(0, matrix.shape[0]):
        min_value = np.min(matrix[row, :])
        index = np.argmin(matrix[row, :])
        if min_value <= th:
            assignment_index.append(index)
        else:
            assignment_index.append(None)

    final_prediction = []
    final_grountruth = []
    for index in range(0, len(assignment_index)):
        if assignment_index[index] is not None:
            final_prediction.append(predicted[index * 2][1])
            final_grountruth.append(groundtruth[assignment_index[index] * 2][1])

    return final_grountruth, final_prediction, assignment_index


def evaluate_system(groundtruth, predicted):
    """
    Evaluate the whole system
    """
    m_groundtruth, m_prediction, assignment = event_assignment(groundtruth, predicted)

    # Metrics
    TP = 0
    FN = 0
    FP = 0

    for index in range(0, len(groundtruth)//2, 1):
        if index in assignment:
            TP += 1
        else:
            FN += 1

    FP = assignment.count(None)

    if (TP == 0 and FP == 0) or (TP == 0 and FN == 0):
        precision = 0
        recall = 0
        fscore = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    if precision != 0 and recall != 0:
        fscore = 2 * ((precision * recall) / (precision + recall))
    else:
        fscore = 0

    return precision, recall, fscore


def evaluate_classificator(groundtruth, predicted):
    """
    Evaluate the classificator interface
    """
    m_groundtruth, m_prediction, assignment = event_assignment(groundtruth, predicted)

    report = classification_report(m_groundtruth, m_prediction, output_dict=True)
    cm = confusion_matrix(m_groundtruth, m_prediction)

    return report, cm


def evaluate_activity_detection(groundtruth, predicted):
    """
    Evaluate activity detection interface (Precision, Recall, f1-score, accuracy)
    """

    # Metrics
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for index in range(0, len(groundtruth), 1):
        if groundtruth[index] == predicted[index] and predicted[index] == 1:
            TP += 1
        elif groundtruth[index] == predicted[index] and predicted[index] == 0:
            TN += 1
        elif groundtruth[index] != predicted[index] and predicted[index] == 0:
            FN += 1
        elif groundtruth[index] != predicted[index] and predicted[index] == 1:
            FP += 1

    if (TP == 0 and FP == 0) or (TP == 0 and FN == 0):
        precision = 0
        recall = 0
        f1_score = 0
        accuracy = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        accuracy = (TP + TN) / (TP + TN + FP + FN)

    # print("Precision " + str(precision))
    # print("Recall " + str(recall))
    # print("F1_score " + str(f1_score))
    # print("Accuracy " +str(accuracy))

    return precision, recall, f1_score, accuracy
    # plot_ad_evaluation(signal,groundtruth,predicted,sr)
