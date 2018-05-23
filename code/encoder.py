#!/usr/bin/env python
'''
Implements LabelEncoder; a label to label id mapping
'''
__author__ = "Sophia Zheng, Rish Gupta, and Manish Pandit"

import os
import json
import operator
from config import labels_file

class LabelEncoder:
    ''' 
    labels to/from id mapper. 
    '''

    def __init__(self):
        ''' initialize dict and id counter '''
        self.label_map = {}
        self.next_id = 0
    
    def add(self, label):
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

    def encode(self, label):
        ''' retrieve the id of the label '''
        return self.label_map.get(label, -1)

    def decode(self, id):
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