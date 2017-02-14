# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:41:21 2016

@author: Hakozaki
"""

import _pickle as cPickle
import numpy as np
import gzip

import sys
import os

import collections

def unpickle(file):
    fo = open(file, 'rb')
    if sys.version_info.major == 2:
        dict = cPickle.load(fo)
    if sys.version_info.major == 3:
        dict = cPickle.load(fo,encoding='latin-1')
    fo.close()
    
    return dict

class DATASET:
    def __init__(self,images,labels):
        self._index_in_epoch = 0
        self._num_examples = 50000
        self._images = images
        self._labels = labels
        
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end]

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
   
dataset_dir = "model/dataset"
img_size = 784

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    
    data.astype(np.float32)
    data = data / 255.0
    
    print("Done")
    
    return data

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    print(labels.size)
    
    #change one-hot vector
    T = np.zeros((labels.size, 10))
    for idx, row in enumerate(T):
        row[labels[idx]] = 1
            
    return T

def load_data():
    train_img = _load_img('train-images-idx3-ubyte.gz')
    train_label =  _load_label('train-labels-idx1-ubyte.gz')
    test_img = _load_img('t10k-images-idx3-ubyte.gz')
    test_label = _load_label('t10k-labels-idx1-ubyte.gz')
    
    train = DATASET(train_img,train_label)
    test = DATASET(test_img,test_label)
    
    Datasets = collections.namedtuple('Datasets',['train','test'])

    return Datasets(train,test)