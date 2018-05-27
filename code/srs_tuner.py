#!/usr/bin/env python
'''
Hyper parameter tuner and retriver.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import voxforge
from srs_builder import ModelBuilder
from config import hp_tuner_in, hp_tuner_out
from config import epochs, batch_size

class Tuner:
    ''' hyper parameter tuner '''
    def __init__(self):      
        pass
    
    def tune(self):
        # get all the data from files as numpy arrays
        # Note: KerasClassifier has issues supporting h5 files so 
        # in-memory representation is required.
        X_Train, Y_Train, X_Dev, Y_Dev, X_Test, Y_Test = voxforge.data_from_files()
        num_speakers = Y_Train.shape[1]
        
        # GridSearchCV requires Y to be flat so reverse the one-hot.
        Y_Train = np.argmax(Y_Train, axis=1)

        # classifier wrapper around the model 
        model = KerasClassifier(build_fn=ModelBuilder(X_Train[0].shape, num_speakers),
                                epochs=epochs, 
                                batch_size=batch_size,
                                verbose=0)

        # prepare the grid
        with open(hp_tuner_in, "r") as fp:
            param_grid = json.load(fp)
        
        # grid search for best hyper parameters
        grid = GridSearchCV(model,
                            param_grid=param_grid,
                            return_train_score=True,
                            scoring=['precision_macro','recall_macro','f1_macro'],
                            refit='precision_macro')

        # fit the training data
        grid_results = grid.fit(X_Train, Y_Train)
        print('Parameters of the best model: ')
        print(grid_results.best_params_)

        # save the best hyper parameters
        with open(hp_tuner_out, 'w') as fp:
            json.dump(grid_results.best_params_, fp, indent=4)
        return grid_results.best_params_

def main():
    Tuner().tune()

if __name__ == '__main__':
    main()