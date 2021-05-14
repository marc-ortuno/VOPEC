from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from evaluate_classificator import get_prediction_time_instants
from models import Waveform
from utils import load_groundtruth, read_csv, plot_metrics_classification_boxplot
from utils import plot_audio, plot_odf, plot_confusion_matrix, plot_evaluation_report
from evaluate import evaluate_system
import pickle
import pandas as pd

# Import model
filename = './app/finalized_model_mfccs.sav'
knn_model = pickle.load(open(filename, 'rb'))

model_normalization = './app/model_normalization_mfccs.csv'

normalization_values = pd.read_csv(model_normalization)
# ../../RawDataset/LML_1617020140983/Kick_LML
path = '../../RawDataset/MAF/MafP2'

audio = Waveform(path=path + ".wav")
groundtruth = read_csv(path + ".csv")

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

precision, recall, fscore, cm, fm = evaluate_system(groundtruth, prediction)
plot_metrics_classification_boxplot(precision, recall, fscore, "mfcc")
# EVALUATION METRICS PLOTS
#plot_evaluation_report(report)
# plot_confusion_matrix(cm)
