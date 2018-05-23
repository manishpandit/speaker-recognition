#!/usr/bin/env python
'''
defines the CNN model
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def model(input_shape, num_categories):
    # sequential model
    model = Sequential()
    # Convolution layer: 3X3 filter, 32 filters, relu
    model.add(Conv2D(32, kernel_size=(3, 3),  padding='same', 
        activation='relu', input_shape=input_shape))
    # Max pool later 2, 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Batch norm
    model.add(BatchNormalization())
    # Convolution layer: 3X3 filter, 64 filters, relu
    model.add(Conv2D(64, kernel_size=(3, 3),  padding='same', 
        activation='relu'))
    # Max pool later 2, 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # dropout for regularization
    model.add(Dropout(0.20))
    # Flatten
    model.add(Flatten())
    # Batch norm
    model.add(BatchNormalization())
    # Dense 2048, relu
    model.add(Dense(2048, activation='relu'))
    # dropout for regularization
    model.add(Dropout(0.20))
    # Batch norm
    model.add(BatchNormalization())
    # Softmax with number of distinct labels
    model.add(Dense(num_categories, activation='softmax'))

    return model