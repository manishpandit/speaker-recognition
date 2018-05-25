#!/usr/bin/env python
'''
provide access to model's hyper parameters. 
If best parameters are saved in a file from tuning, it uses that;
else it uses defaults from config.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import json
from config import ACTIVATION, OPTIMIZER, DROP_OUT, BATCH_SIZE, EPOCHS
from config import model_hyper_params

class HyperParams:
    ''' hyper parameter tuning and retrival '''
    def __init__(self):
        ''' 
        reads hyper parameters from hyper parameter file if exists
        or from default values in config
        '''
        hp = {}
        if os.path.exists(model_hyper_params):
            with open(model_hyper_params, 'r') as fp:
                hp = json.load(fp)

        self.activation = hp.get('activation', ACTIVATION)
        self.optimizer = hp.get('optimizer', OPTIMIZER)
        self.dropout = hp.get('dropout', DROP_OUT)
        self.batchsize = hp.get('batch_size', BATCH_SIZE)
        self.epochs = hp.get('epochs', EPOCHS)
    
    def activation(self):
        ''' returns best or default activation '''
        return self.activation
    
    def optimizer(self):
        ''' returns best or default optimizer '''
        return self.optimizer
    
    def dropout(self):
        ''' returns best or default dropout rate '''
        return self.dropout
    
    def batchsize(self):
        ''' returns best or default batchsize '''
        return self.batchsize

    def epochs(self):
        ''' returns best or default epochs '''
        return self.epochs