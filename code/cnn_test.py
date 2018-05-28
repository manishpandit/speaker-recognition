#!/usr/bin/env python
'''
train text independent speaker recognition convolutional neural network.
save model and parameters.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import keras
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import cm
from python_speech_features import mfcc
from python_speech_features import logfbank
import voxforge
from encoder import LabelEncoder
from config import quick_test_dir, max_pad_len, reports_dir
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
    fbank = voxforge.wav2fbank(file_path, max_pad_len)
    fbank = fbank.reshape(1, max_pad_len, 26, 1)
    label_id = np.argmax(model.predict(fbank))
    print("Audio file {0}, predicted speaker: {1}".format(f, encoder.decode(label_id)))

    sample_rate, samples = wav.read(os.path.join(quick_test_dir, f))
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    mfcc_feat = mfcc(samples, sample_rate)
    fbank_feat = logfbank(samples, sample_rate)

    plt.figure(1)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(os.path.join(reports_dir, f + '_spectro.png'))

    plt.figure(2)
    ig, ax = plt.subplots()
    mfcc_data = np.swapaxes(mfcc_feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    #Showing mfcc_feat
    plt.plot(mfcc_feat)    
    plt.savefig(os.path.join(reports_dir, f + '_mfcc_feat.png'))

    plt.figure(3)
    ig, ax = plt.subplots()
    fbank_data = np.swapaxes(fbank_feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('Filter Bank')
    #Showing mfcc_feat
    plt.plot(fbank_feat)    
    plt.savefig(os.path.join(reports_dir, f + '_fbank_feat.png'))
