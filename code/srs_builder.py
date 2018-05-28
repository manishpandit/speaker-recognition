#!/usr/bin/env python
'''
Wrapper class around speaker regocnition cnn model.
provides utilities to save and load trained models.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from config import model_file, model_params
from config import activation, optimizer, dropout_rate, lambda_regularizer


class ModelBuilder:
    ''' builds a convolutional nueral network model '''

    def __init__(self, input_shape, num_categories):
        ''' 
        constructor 
        input_shape: shape of X.  This is required for the input layer definition.
        num_categories: number of unique softmax output values expected. 
        In this case, number of unique speakers in the dataset.
        activation: activation function to use for all layers except the output layer.
            valid values are: 'relu' and 'tanh'
        optimizer: optimizer to use for gradient descent.
            valid values are: 'rmsprop', 'adam' and 'adadelta'
        dropout: dropout rate.
        The default values are read from config.
        '''
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.activation = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        
    def __call__(self, activation=activation, optimizer=optimizer, dropout_rate=dropout_rate):
        ''' 
        default method on this object. It was designed to support KerasClassifier definition.
        '''

        self.activation = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate

        # sequential model
        model = Sequential()
        # Convolution layer: 3X3 filter, 32 filters
        model.add(Conv2D(4, kernel_size=(3, 3),  padding='same', 
            activation=self.activation,
            input_shape=self.input_shape))
        # Max pool later 2, 2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        # Batch norm
        model.add(BatchNormalization())
        # Convolution layer: 3X3 filter, 64 filters
        model.add(Conv2D(8, kernel_size=(3, 3),  padding='same', 
            activation=self.activation))
        # Max pool later 2, 2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        # Batch norm
        model.add(BatchNormalization())
        # Convolution layer: 3X3 filter, 128 filters
        model.add(Conv2D(16, kernel_size=(3, 3),  padding='same', 
            activation=self.activation))
        # Max pool later 2, 2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        # Batch norm
        model.add(BatchNormalization())
        
        # Flatten
        model.add(Flatten())
        # dropout for regularization
        model.add(Dropout(self.dropout_rate))
        # Dense layer
        model.add(Dense(self.num_categories * 2, 
            activation=activation,
            kernel_regularizer=regularizers.l2(lambda_regularizer)))
        # Batch norm
        model.add(BatchNormalization())
        # dropout for regularization
        model.add(Dropout(self.dropout_rate))

        # Softmax with number of distinct labels
        model.add(Dense(self.num_categories, activation='softmax'))

        # Compile the model with categorial cross entropy loss
        # Adadelta optimizer
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optimizer_instance(),
              metrics=['acc'])

        return model

    def optimizer_instance(self):
        opt = None
        if self.optimizer == 'rmsprop':
            opt = keras.optimizers.rmsprop()
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam()
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta()
        return opt

    def save(self, model):
        ''' serialize model and parameters ''' 
        if not os.path.exists(os.path.dirname(model_file)):
            os.makedirs(os.path.dirname(model_file))
            
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(model_params)
        print("Saved model to disk")
    
    def load(self):
        ''' serialize model and parameters ''' 
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_params)
        print("Loaded model from disk")
        # compile
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=self.optimizer_instance(),
              metrics=['acc'])
        return model
