# Vocal Percussion Classification for Real Time Context

A Python prototype capable of classifying a stream of vocalized percussion (beatbox) in real-time.

# Usage

Go to Prototype folder and activate the virtual environment
`source env/bin/activate`

:warning: [Librosa's library](https://librosa.org/doc/latest/index.html) can give errors, make sure you have the dependency properly installed. :warning:

# Scripts

[`run.py`](./Prototype/core/run.py): This script is used to run an off-line simulation of the system in order to evaluate some interfaces.


[`rt_test.py`](./Prototype/core/rt_test.py): This script executes a real time demonstration of the prototype. It uses [sounddevice](https://python-sounddevice.readthedocs.io/) to obtain streams of a microphone input. On each callback we execute de process of the prototype.

# Evaluation scripts

[`evaluate_activity_detection.py`](./Prototype/core/evaluate_activity_detection.py): The purpose of this script is to obtain the **precision**, **recall**, f-**score** and **accuracy** metrics by comparing the activity detection output with the corresponding groundtruth (annotation file).


[`evaluate_classificator.py`](./Prototype/core/evaluate_classificator.py): Used to obtain the classification report and confusion matrix from the beatbox classification interface.


[`evaluate_system.py`](./Prototype/core/evaluate_system.py): Used to obtain precision, recall, f-score and accuracy metrics of the the system event detection by combining the activity detection system with the different classification models.

# Train model scripts

[`train_model_mfccs.py`](./Prototype/core/train_model_mfccs.py): Creates a model in sav format based on mfcc features.


[`train_model_all_features.py`](./Prototype/core/train_model_all_features.py): Creates a model in sav format based on a set of features from [librosa](https://librosa.org/doc/main/feature.html).
