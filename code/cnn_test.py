#!/usr/bin/env python
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
import numpy as np
import voxforge
from encoder import LabelEncoder
from config import quick_test_dir, MAX_PAD_LEN
from srs_builder import ModelBuilder

# load test data
X, Y = voxforge.get_test_data()
# load labels
encoder = LabelEncoder()
encoder.load()

builder = ModelBuilder(X[0].shape, Y.shape[1])
model = builder.load()
 
# evaluate loaded model on test data
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# perform quick tests
for f in os.listdir(quick_test_dir):
    if not f.endswith('.wav'):
        continue
    file_path = os.path.join(quick_test_dir, f)
    mfcc = voxforge.wav2mfcc(file_path, MAX_PAD_LEN)
    mfcc = mfcc.reshape(1, 20, MAX_PAD_LEN, 1)
    label_id = np.argmax(model.predict(mfcc))
    print("Audio file {0}, predicted speaker: {1}".format(f, encoder.decode(label_id)))
