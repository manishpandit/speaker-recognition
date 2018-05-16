#!/usr/bin/env python
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import voxforge
from config import model_file, model_params, model_plot

# get training and dev data
X_Train, Y_Train = voxforge.get_train_data()
X_Dev, Y_Dev = voxforge.get_dev_data()
labels = voxforge.Labels()
labels.load()
num_speakers = labels.len()

#################################### CNN model definition #################################
# sequential model
model = Sequential()
# Convolution layer: 2X2 filter, 32 filters, relu
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 256, 1)))
# Max pool later 2, 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# 25% dropout for regularization
model.add(Dropout(0.25))
# Flatten
model.add(Flatten())
# Batch norm
model.add(BatchNormalization())
# Dense 128, relu
model.add(Dense(128, activation='relu'))
# 25% drop out
model.add(Dropout(0.25))
# Batch norm
model.add(BatchNormalization())
# Softmax with number of distinct labels
model.add(Dense(num_speakers, activation='softmax'))

# print the model
print(model.summary())

# save the visual description of the model
plot_model(model, to_file=model_plot, show_shapes=True, show_layer_names=True)

# Compile the model with categorial cross entropy loss
# Adadelta optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

############################# End CNN model definition #################################

# Run the optimization process and validate against dev set
# Note: We are reading data in batch mode from h5 file so shuffle 
# parameter is required.
model.fit(X_Train, Y_Train,
    batch_size = 64, epochs = 10, verbose = 1, 
    validation_data = (X_Dev, Y_Dev), shuffle = 'batch')

# serialize model to JSON
if not os.path.exists(os.path.dirname(model_file)):
    os.makedirs(os.path.dirname(model_file))
    
model_json = model.to_json()
with open(model_file, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_params)
print("Saved model to disk")
