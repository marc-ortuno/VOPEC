from interfaces import train_model
from dataset import get_dataset, dataset_analyzer
import pickle 

'''
This script aims to generate a kNN classifier model.
'''

data = get_dataset() 
dataset_analyzer(data) #Analyze dataset and generate metrics plot

knn_model = train_model(data) #Train the model

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(knn_model, open(filename, 'wb'))