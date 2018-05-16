#!/usr/bin/env python
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
from keras.models import model_from_json
import numpy as np
import voxforge
from config import model_file, model_params, quick_test_dir

# load test data
X, Y = voxforge.get_test_data()
# load labels
labels = voxforge.Labels()
labels.load()

# load json and create model
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_params)
print("Loaded model from disk")
 
# evaluate loaded model on test data
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# perform quick tests
for f in os.listdir(quick_test_dir):
    if not f.endswith('.wav'):
        continue
    file_path = os.path.join(quick_test_dir, f)
    mfcc = voxforge.wav2mfcc(file_path)
    mfcc = mfcc.reshape(1, 20, 256, 1)
    label_id = np.argmax(model.predict(mfcc))
    print("Audio file {0}, predicted speaker: {1}".format(f, labels.get_label(label_id)))
