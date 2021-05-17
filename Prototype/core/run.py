import numpy as np

from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from models import Waveform
from utils import load_groundtruth, read_csv, plot_metrics_classification_boxplot
from utils import plot_audio, plot_odf, plot_confusion_matrix, plot_evaluation_report
from evaluate import evaluate_system, evaluate_activity_detection, evaluate_classificator
import pickle
import pandas as pd

"""
The run script is used to run an off-line simulation of the system in order to evaluate some interfaces 
for performance testing.
"""
# Import model
from utils.load_csv import get_prediction_time_instants, load_annotation

filename = './app/finalized_model_mfccs.sav'
knn_model = pickle.load(open(filename, 'rb'))

model_normalization = './app/model_normalization_mfccs.csv'

normalization_values = pd.read_csv(model_normalization)
# ../../RawDataset/LML_1617020140983/Kick_LML
path = '../../RawDataset/MOB_1616751225808/Snare_MOB'

audio = Waveform(path=path + ".wav")
groundtruth = load_annotation(path + ".csv")



# Init system
init_pre_processing()
init_activity_detection(func_type=1)
init_feature_extraction(func_type="mfcc", by_pass=False, n_mfcc_arg=20, norm_file=normalization_values)
init_classificator(knn_model=knn_model, by_pass=False)
buffer_len = 512

# Call system
result = main(audio, buffer_len)

prediction = get_prediction_time_instants(result['ONSET_LOCATIONS'], result['PREDICTION'], audio.sample_rate)
# Plot results
plot_audio(audio.waveform, result['SIGNAL_PROCESSED'], audio.sample_rate)
plot_odf(audio.filename, audio.waveform, result['SIGNAL_PROCESSED'], audio.sample_rate, result['ONSET_LOCATIONS'],
         result['HFC'], result['THRESHOLD'])

groundtruth_activity = np.zeros(len(result['ONSET_LOCATIONS']))
# Transform annotation in the desired format (1 activity, 0 non-activity)
for i in range(0, len(groundtruth), 2):
    sample_instant_1 = int(float(groundtruth[i][0]) * audio.sample_rate)
    sample_instant_2 = int(float(groundtruth[i + 1][0]) * audio.sample_rate)
    groundtruth_activity[sample_instant_1:sample_instant_2] = 1

# Evaluation
precision_ad, recall_ad, fscore_ad, accuracy_ad = evaluate_activity_detection(groundtruth_activity,
                                                                              result['ONSET_LOCATIONS'])

report, cm = evaluate_classificator(groundtruth, prediction)
precision, recall, fscore = evaluate_system(groundtruth, prediction)

print('----------------------------------------------------------------------')
print('Prototype report')
print('----------------------------------------------------------------------')
print('Activity detection evaluation:')
print('\n Precision:' + str(precision_ad) +
      '\n Recall:' + str(recall_ad) +
      '\n F1-score:' + str(fscore_ad) +
      '\n Accuracy:' + str(accuracy_ad)
      )
print('----------------------------------------------------------------------')
print('Classification evaluation:')
plot_evaluation_report(report)
if len(cm) == 3:
    plot_confusion_matrix(cm)
print('----------------------------------------------------------------------')
print('System evaluation:')
print('\n Precision:' + str(precision) +
      '\n Recall:' + str(recall) +
      '\n F1-score:' + str(fscore)
      )
print('----------------------------------------------------------------------')

