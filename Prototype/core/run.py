from app import main, init_activity_detection, init_classificator, init_feature_extraction, init_pre_processing
from models import Waveform
from utils import load_groundtruth
from utils import plot_audio, plot_odf, plot_confusion_matrix, plot_evaluation_report
from evaluate import evaluate_system
import pickle

# Import model
filename = './app/finalized_model_v2.sav'
knn_model = pickle.load(open(filename, 'rb'))

# ../../RawDataset/LML_1617020140983/Kick_LML
path = './data/realtime_demo__vh7bpu0'

audio = Waveform(path=path + ".wav")
groundtruth = load_groundtruth('../../RawDataset/LML_1617020140983/Kick_LML' + ".csv")

# Init system
init_pre_processing()
init_activity_detection(func_type=1)
init_feature_extraction(by_pass=False, n_mfcc_arg=10)
init_classificator(knn_model=knn_model, by_pass=False)
buffer_len = 512

# Call system
result = main(audio, buffer_len)
# Plot results
plot_audio(audio.waveform, result['SIGNAL_PROCESSED'], audio.sample_rate)
plot_odf(audio.filename, audio.waveform, result['SIGNAL_PROCESSED'], audio.sample_rate, result['ONSET_LOCATIONS'],
         result['HFC'], result['THRESHOLD'])
report, cm = evaluate_system(groundtruth, result['PREDICTION'])

# EVALUATION METRICS PLOTS
plot_evaluation_report(report)
plot_confusion_matrix(cm)
