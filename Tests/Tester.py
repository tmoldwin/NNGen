# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:23:44 2015

@author: tmold_000
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from pandas import *

def fun1(X):
    return sin(X[0]**2) + X[1]**2 + multiply(X[0],X[1])
    
def fun2(X):
    return exp(3*X[0])+tan(4**X[1])
    
def fun3(X):
    return cos(X[0]*X[1])**(X[1]+X[0])
    
def fun4(X):
    return 3*X[0]**2+X[1]**3
    
def makeSets(X,Y, trainingPercent):
    cutoffIndex = len(X)*(1-trainingPercent)
    xTraining = X[0:cutoffIndex,:]
    yTraining = Y[0:cutoffIndex,:]
    xTest = X[cutoffIndex:len(X),:]
    yTest = Y[cutoffIndex:len(Y),:]
    return xTraining, yTraining, xTest, yTest

numExamples = 75000
trainingPercent = 0.75
X = np.random.rand(numExamples, 2)
df = DataFrame(X)

Y = np.empty([numExamples,4])
Y[:,0] = df.apply(fun1, axis = 1)
Y[:,1] = df.apply(fun2, axis = 1)
Y[:,2] = df.apply(fun3, axis = 1)
Y[:,3] = df.apply(fun4, axis = 1)


xTraining, yTraining, xTest, yTest = makeSets(X, Y, trainingPercent)

xTraining = array(xTraining).astype(float32)
yTraining = array(yTraining).astype(float32)
xTest = array(xTest).astype(float32)
yTest = array(yTest).astype(float32)

net1 = NeuralNet(
    layers=[  # four layers: two hidden layers
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
     #  ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 2),  # 2 input
    hidden_num_units=400,  # number of units in hidden layer
 #   hidden1_num_units=400,
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=4,  # 4 outputs

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=1000,  # we want to train this many epochs
    verbose=1,
    )


net1.fit(minMaxNormalize(xTraining), gaussNormalize(yTraining)) # This thing try to do the fit itself



y_pred = net1.predict(minMaxNormalize(xTest))  # Here I test the other 30k random samples
error = gaussNormalize(yTest)-y_pred
print(error)