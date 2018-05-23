#!/usr/bin/env python
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
from keras.utils import to_categorical
import numpy as np
import voxforge
from config import model_file, model_params, MAX_PAD_LEN, labels_file, EPOCH
from reports import plot_history
import cnn_model
from encoder import LabelEncoder

# get training and dev data
X_Train, Y_Train = voxforge.get_train_data()
X_Dev, Y_Dev = voxforge.get_dev_data()
encoder = LabelEncoder()
encoder.load()
num_speakers = encoder.len()

model = cnn_model.model(X_Train[0].shape, num_speakers)

# print the model
print(model.summary())

# Compile the model with categorial cross entropy loss
# Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['acc'])

# Run the optimization process and validate against dev set
# Note: We are reading data in batch mode from h5 file so shuffle 
# parameter is required.
history = model.fit(X_Train, Y_Train,
    batch_size = 128, epochs = EPOCH, verbose = 1, 
    validation_data = (X_Dev, Y_Dev), shuffle = 'batch')

# plot history
plot_history(history)

# serialize model to JSON
if not os.path.exists(os.path.dirname(model_file)):
    os.makedirs(os.path.dirname(model_file))
    
model_json = model.to_json()
with open(model_file, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_params)
print("Saved model to disk")
