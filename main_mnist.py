#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:55:09 2017

@author: hakozaki
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *

from model.mnist import load_data

class TwoLayerNet:
    def __init__(self):
        self.W1 = np.random.randn(784,100)
        self.W2 = np.random.randn(100,10)
        self.b1 = np.zeros(100)
        self.b2 = np.zeros(10)
        
    def predict(self,X):
        z1 = np.dot(X,self.W1) + self.b1
        u1 = 1 / (1 + np.exp(-z1)) 
        z2 = np.dot(u1,self.W2) + self.b2
        y = self.softmax(z2)
        return u1,y
        
    def loss(self,x,d):
        u1 , y = self.predict(x)
        return self.cross_entropy_error(y , d)
        
    def train(self,x,d):
        
        u1 , y = self.predict(x)
        
        # バッチ処理
        N = x.shape[0]

        eta = 0.5

        delta_0 = y - d        

        output_delta = eta * ( 1 / N ) * np.dot(u1.T , delta_0)
        output_bias = eta * ( 1 / N) * np.sum(delta_0,axis = 0)

        self.W2 = self.W2 - output_delta
        self.b2 = self.b2 - output_bias
        
        f_u = (1 - u1) * u1
        before_weight = np.dot(y - d,net.W2.T)
        
        delta_1 = f_u * before_weight
        
        hidden_delta = eta * (1 / N) * np.dot(x.T, delta_1)
        hidden_bias = eta * (1 / N) * np.sum(delta_1,axis = 0)
        
        self.W1 = self.W1 - hidden_delta
        self.b1 = self.b1 - hidden_bias
        
    def softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))
        
    def cross_entropy_error(self,y, d):
        return -np.sum(d * np.log(y))/y.shape[0]

    def mean_squared_error(self,y,d):
        e = ((y - d) * (y - d)) / 2
        return np.sum(e,axis = 1)
        
    def accuracy(self, x, t):
        u1 , y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def relu(self,x):
        return np.maximum(0, x)


    
if __name__ == '__main__':
    
    net = TwoLayerNet()
    
    mnist = load_data()
    
    loss_data = []

    print(net.accuracy(mnist.test.images,mnist.test.labels))

    epoch = 2000
    
    for i in range(epoch):
        batch_x , batch_y =  mnist.train.next_batch(50)
        
        net.loss(batch_x,batch_y)
        
        net.train(batch_x,batch_y)
        
        loss_data.append(net.loss(batch_x,batch_y))
    
    
    x = np.arange(0,epoch,1)
    plt.plot(x,loss_data)
    
    print(net.accuracy(mnist.test.images,mnist.test.labels))
    
    
    
    
    
    
   
    
    