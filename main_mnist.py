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

class MultiLayerNet:
    def __init__(self):
        self.W1 = np.random.randn(784,128)
        self.W2 = np.random.randn(128,64)
        self.W3 = np.random.randn(64,10)
        self.b1 = np.zeros(128)
        self.b2 = np.zeros(64)
        self.b3 = np.zeros(10)
        self.loss_data = []
        
    def predict(self,X):
        z1 = np.dot(X,self.W1) + self.b1
        u1 = 1 / (1 + np.exp(-z1)) 
        z2 = np.dot(u1,self.W2) + self.b2
        u2 = 1 / (1 + np.exp(-z2))
        z3 = np.dot(u2,self.W3) + self.b3
        y = self.softmax(z3)
        return u1,u2,y
        
    def loss(self,x,d):
        u1, u2, y = self.predict(x)
        return self.cross_entropy_error(y , d)
        
    def train(self,x,d):
        
        self.loss_data.append(self.loss(batch_x,batch_y))
        
        u1, u2, y = self.predict(x)
        
        # batch 
        N = x.shape[0]
        eta = 0.5
        
        #third layer
        delta_3 = y - d        

        output_delta = eta * ( 1 / N ) * np.dot(u2.T , delta_3)
        output_bias = eta * ( 1 / N) * np.sum(delta_3,axis = 0)

        self.W3 = self.W3 - output_delta
        self.b3 = self.b3 - output_bias
        
        #second layer
        f_u2 = (1 - u2) * u2
        before_weight2 = np.dot(y - d,net.W3.T)
        
        delta_2 = f_u2 * before_weight2
        
        hidden_delta2 = eta * (1 / N) * np.dot(u1.T, delta_2)
        hidden_bias2 = eta * (1 / N) * np.sum(delta_2,axis = 0)
        
        self.W2 = self.W2 - hidden_delta2
        self.b2 = self.b2 - hidden_bias2
        
        #first layer
        f_u1 = (1 - u1) * u1
        before_weight1 = np.dot(delta_2,net.W2.T)
        
        delta_1 = f_u1 * before_weight1
        
        hidden_delta1 = eta * (1 / N) * np.dot(x.T, delta_1)
        hidden_bias1 = eta * (1 / N) * np.sum(delta_1,axis = 0)
        
        self.W1 = self.W1 - hidden_delta1
        self.b1 = self.b1 - hidden_bias1
        
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
        u1, u2, y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def relu(self,x):
        return np.maximum(0, x)
        
    def error_gragh(self):
        x = np.arange(0,np.size(self.loss_data),1)
        plt.plot(x,self.loss_data)
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.ylim([0,3])

    
if __name__ == '__main__':
    
    net = MultiLayerNet()
    
    mnist = load_data()
    
    print("Before Accuracy : %f" % net.accuracy(mnist.test.images,mnist.test.labels))

    epoch = 5000
    
    for i in range(epoch):
        batch_x , batch_y =  mnist.train.next_batch(50)
        
        net.train(batch_x,batch_y)
        
    net.error_gragh()
    
    print("After Accuracy : %f" % net.accuracy(mnist.test.images,mnist.test.labels))
    
    
    
    
    
    
   
    
    