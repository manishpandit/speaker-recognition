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

# tuner dir
tuner_dir = os.path.join(res_dir, "tuner")

# input file for hyper parameters tuner (json format)
hp_tuner_in = os.path.join(tuner_dir, "hp_tuner_in.json")

# output file of hyper parameters tuner (json format)
hp_tuner_out = os.path.join(tuner_dir, "hp_tuner_out.json")

# dir where results are stored
reports_dir = os.path.join(model_dir, "reports")

# quick test dir
quick_test_dir = os.path.join(res_dir, "quick_test")

# MFCC max_pad length
max_pad_len = 196

# Default activation: valid set: 'relu' and 'tanh'
activation = 'relu'

# Default optimizer: valid set: 'adadelta', 'adam' and 'rmsprop'
optimizer = 'adam'

# Default Number of training epochs
epochs = 32

# Default batch size
batch_size = 64

# Default dropout rate
dropout_rate = 0.25

# lamda value of the regulizer. set it to zero for no regularizarion.
lambda_regularizer = 0.005