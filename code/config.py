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
raw_data_dir = os.path.join(res_dir, "raw_data")

# dir where h5 file created from converting wav files
data_dir = os.path.join(res_dir, "data")

# name of the h5 data file
data_file = os.path.join(data_dir, "voxforge.h5") 

# dir where model and parameters is stored
model_dir = os.path.join(res_dir, "model")

# model file (json format)
model_file = os.path.join(model_dir, "model.json")

# model parameters (h5 format)
model_params = os.path.join(model_dir, "model.h5")

# model parameters (h5 format)
model_plot = os.path.join(model_dir, "model_plot.png")

# label mapping file
labels_file = os.path.join(data_dir, "labels.json")

# quick test dir
quick_test_dir = os.path.join(res_dir, "quick_test")

