'''
stores application wide configurations. 
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os

# application root directory (relative to the /code dir) 
project_root = os.path.pardir

# resources dir
res_dir = os.path.join(project_root, "res")

# voxforge dataset root dir
raw_data_dir = os.path.join(res_dir, "voxforge/archive")
#raw_data_dir = os.path.join(res_dir, "voxforge_mini/archive")

# dir where h5 file created from converting wav files
data_dir = os.path.join(res_dir, "data")

# name of the h5 data file
data_file = os.path.join(data_dir, "voxforge.h5")
#data_file = os.path.join(data_dir, "voxforge_mini.h5") 

# label mapping file
labels_file = os.path.join(data_dir, "labels.json")

# dir where model and parameters is stored
model_dir = os.path.join(res_dir, "model")

# model file (json format)
model_file = os.path.join(model_dir, "model.json")

# model parameters (h5 format)
model_params = os.path.join(model_dir, "model.h5")

# model hyper parameters (json format)
model_hyper_params = os.path.join(model_dir, "model_hp.json")

# tuner file (json format)
model_tuner_file = os.path.join(res_dir, "tune/model_tuner.json")

# quick test dir
quick_test_dir = os.path.join(res_dir, "quick_test")

# MFCC max_pad length
MAX_PAD_LEN = 196

# Default activation
ACTIVATION = 'relu'

# Default optimizer 
OPTIMIZER = 'adadelta'

# Default Number of training epochs
EPOCHS = 5

# Default batch size
BATCH_SIZE = 128

# Default dropout
DROP_OUT = 0.25