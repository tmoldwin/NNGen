# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 17:38:42 2015

@author: tmold_000
"""

import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import Normalizers
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from pandas import *

def makeSets(X,Y, trainingPercent):
    cutoffIndex = len(X)*trainingPercent
    xTraining = X[0:cutoffIndex,:]
    yTraining = Y[0:cutoffIndex,:]
    xTest = X[cutoffIndex:len(X),:]
    yTest = Y[cutoffIndex:len(Y),:]
    return xTraining, yTraining, xTest, yTest

def fun1(X):
    return 5 * X**2 + 2 * X
    

numExamples = 50000
trainingPercent = 0.75
X = np.random.rand(numExamples, 1)
df = DataFrame(X)
Y = np.empty([numExamples,1])
Y[:] = df.apply(fun1)



xTraining, yTraining, xTest, yTest = makeSets(X, Y, trainingPercent)

xTraining = array(xTraining).astype(float32)
yTraining = array(yTraining).astype(float32)
xTest = array(xTest).astype(float32)
yTest = array(yTest).astype(float32)

numInputs = 1
numOutputs = 1



X = Normalizers.minMaxNormalize(xTraining)
Xtest = Normalizers.minMaxNormalize(xTest)

#Y = Normalizers.gaussNormalize(trainingOutputs)
#Ytest = Normalizers.gaussNormalize(testOutputs)
#
#X = trainingInputs
#Xtest = testInputs
Y = yTraining
Ytest = yTest

hiddenSize = 40;

model = Sequential()
model.add(Dense(numInputs, hiddenSize, init='lecun_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(hiddenSize, hiddenSize, init='lecun_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(hiddenSize, numOutputs, init='lecun_uniform'))
model.add(Activation('linear'))

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(loss='mean_squared_error', optimizer=Adam)
model.fit(X, Y, nb_epoch=1, batch_size=16)
score = model.evaluate(Xtest, Ytest, batch_size=100)


import pylab
# neural net approximation
pylab.close()

pylab.scatter(Xtest,
           model.predict(Xtest), linewidth = 2,
           color = 'blue', label = 'NN output')

# target function
pylab.scatter(Xtest,
           Ytest, linewidth = 2, color = 'red', label = 'target')

pylab.grid()
pylab.legend()
pylab.show()

