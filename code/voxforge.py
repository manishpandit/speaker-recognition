#!/usr/bin/env python
'''
provides functionality to manage audio data.
creates h5 file and labels mapping file from 
voxforge dataset.
converts wav files to mfcc matrix.
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import librosa
import h5py
import json
import operator
import numpy as np
import keras as K
from config import raw_data_dir, data_dir, data_file, labels_file
from config import model_dir, model_file, model_params

###################################### class Labels #######################################
class Labels:
    ''' 
    dataset label and id management 
    '''

    def __init__(self):
        ''' initialize dict and id counter '''
        self.label_map = {}
        self.next_id = 0
    
    def add_label(self, label):
        ''' 
        adds a label to map if not already there.
        returns id of the label.
        '''
        id = self.label_map.get(label, -1)
        if id != -1:
            return id
        self.label_map[label] = self.next_id
        id = self.next_id 
        self.next_id += 1
        return id

    def get_id(self, label):
        ''' retrieve the id of the label '''
        return self.label_map.get(label, -1)

    def get_label(self, id):
        ''' retrieve label from the id '''
        for k, v in self.label_map.items():
            if v == id:
                return k
        return ""
    
    def save(self):
        ''' save the data to a file '''
        if not os.path.exists(os.path.dirname(labels_file)):
            os.makedirs(os.path.dirname(labels_file))
        with open(labels_file, 'w') as fp:
            json.dump(self.label_map, fp, indent=4)

    def load(self):
        ''' load from a file '''
        with open(labels_file, 'r') as fp:
            self.label_map = json.load(fp)
            self.next_id = max(self.label_map.items(),
             key=operator.itemgetter(1))[1] + 1
    
    def len(self):
        ''' returns number of members in the map '''
        return len(self.label_map)
#################################### End class Labels #####################################

def parse_label(dir):
    ''' parse speaker label from file path '''
    i = dir.find('-')
    if i == -1:
        return None
    return dir[:i]

def wav2mfcc(file_path, max_pad_len = 256):
    ''' convert wav file to mfcc matrix '''
    wave, sr = librosa.load(file_path, mono = True, sr = None)
    mfcc = librosa.feature.mfcc(wave, sr = 16000)
    mfcc = mfcc[:, :max_pad_len]
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width = ((0, 0), (0, pad_width)), mode = 'constant')
    return mfcc

def create_h5_file():
    ''' 
    create h5 file with train, dev and test datasets.
    create labels mapping file.
    '''
    labels = Labels()
    mfcc_vectors = []
    label_vectors = []
    
    # walk the data dir looking for wav file
    for dirpath, dirnames, filenames in os.walk(raw_data_dir):
        for filename in [f for f in filenames if f.endswith(".wav")]:
            # get mfcc matrix for this file
            mfcc = wav2mfcc(os.path.join(dirpath, filename))
            # get label for this file
            label = parse_label(os.path.relpath(dirpath, raw_data_dir))
            # convert to label id
            label_id = labels.add_label(label)

            # add mfcc
            mfcc_vectors.append(mfcc)
            # add label
            label_vectors.append(label_id)
    
    # create numpy array for X and Y
    X_All = np.array(mfcc_vectors)
    Y_All = np.array(label_vectors)
    # random shuffle dataset
    # we need each instance of X in 3 dimensional for CNN so reshape X
    # X is now a 4D matrix
    # convert Y to one hot encoding
    permutation = np.random.permutation(X_All.shape[0])
    X_All = X_All[permutation].reshape(X_All.shape[0], 20, 256, 1)
    Y_All = Y_All[permutation].reshape(Y_All.shape[0], -1)
    Y_All = K.utils.to_categorical(Y_All)

    # split the dataset into train, dev and test sets (90%, 5%, 5% approx)
    m = X_All.shape[0]
    idx_train = int(m * 0.9)
    idx_dev = int(m * 0.95)
    # train
    X_Train = X_All[:idx_train]
    Y_Train = Y_All[:idx_train]
    # dev
    X_Dev = X_All[idx_train:idx_dev]
    Y_Dev = Y_All[idx_train:idx_dev]
    # test
    X_Test = X_All[idx_dev:]
    Y_Test = Y_All[idx_dev:]

    # store the datasets to h5 file
    if not os.path.exists(os.path.dirname(data_file)):
        os.makedirs(os.path.dirname(data_file))
    f = h5py.File(data_file, 'w')
    f.create_dataset('X_Train', data=X_Train)
    f.create_dataset('Y_Train', data=Y_Train)
    f.create_dataset('X_Dev', data=X_Dev)
    f.create_dataset('Y_Dev', data=Y_Dev)
    f.create_dataset('X_Test', data=X_Test)
    f.create_dataset('Y_Test', data=Y_Test)
    f.close()

    # save labels
    labels.save()
    
def get_data(X_Set, Y_Set):
    ''' 
    get dataset by name.
    X_Set and Y_Set are name of the dataset
    returns numpy array of dataset stored in
    h5 file
    '''
    f = h5py.File(data_file, 'r')
    return f[X_Set], f[Y_Set]

def get_train_data():
    ''' 
    get training dataset.
    returns numpy array of dataset stored in h5 file.
    '''
    return get_data("X_Train", "Y_Train")

def get_dev_data():
    ''' 
    get dev dataset.
    returns numpy array of dataset stored in h5 file.
    '''
    return get_data("X_Dev", "Y_Dev")

def get_test_data():
    ''' 
    get test dataset.
    returns numpy array of dataset stored in h5 file.
    '''
    return get_data("X_Test", "Y_Test")

def get_speaker_map():
    speaker_map = {}
    for d in os.listdir(raw_data_dir):
        audio_dir = os.path.join(raw_data_dir, d)
        if not os.path.isdir(audio_dir):
            continue
        speaker = parse_label(d)
        if speaker == None:
            continue
        for dirpath, dirnames, filenames in os.walk(audio_dir):
            for filename in [f for f in filenames if f.endswith(".wav")]:
                speaker_files = speaker_map.get(speaker, [])
                speaker_files.append(os.path.join(dirpath, filename))
                speaker_map[speaker] = speaker_files
    return speaker_map

def get_speaker_data(audio_files):
    X = []
    for f in audio_files:
        mfcc = wav2mfcc(f)
        X.append(mfcc)
    return np.array(X)

def main():
        
    create_h5_file()
    X, Y = get_train_data()
    print("X_Train, Y_Train shape:", X.shape, Y.shape)
    X, Y = get_dev_data()
    print("X_Dev, Y_Dev shape:", X.shape, Y.shape)
    X, Y = get_test_data()
    print("X_Test, Y_Test shape:", X.shape, Y.shape)
   

    """ speaker_map = get_speaker_map()
    for k, v in speaker_map.items():
        print("speaker: ", k, end="\t")
        print(get_speaker_data(v).shape) """


if __name__ == '__main__':
    main()
