# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:42:37 2015

@author: tmold_000
"""

import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import dataParser
import Normalizers
import numpy as np
import matplotlib.pyplot as plt




#generations, generationsToInputs, generationsToOutputs = dataParser.parse(fname = "whole_population_0.txt")
numInputs = 31
numOutputs = 28

trainingInputs, trainingOutputs, testInputs, testOutputs = dataParser.makeSets(generationsToInputs, generationsToOutputs, generations[150:200], 50000, 0.25)



X = Normalizers.gaussNormalize(trainingInputs)
Xtest = Normalizers.gaussNormalize(testInputs)
#
#Y = Normalizers.gaussNormalize(trainingOutputs)
#Ytest = Normalizers.gaussNormalize(testOutputs)
#
#X = trainingInputs
#Xtest = testInputs
Y = trainingOutputs
Ytest = testOutputs

hiddenSize = 400;
hiddenSize2 = 50
#winner 400 one layer  dropout 0.3 "weights1" "weights2"
model = Sequential()
model.add(Dense(numInputs, hiddenSize, init='lecun_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
#model.add(Dense(hiddenSize, hiddenSize, init='uniform'))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.1))
#model.add(Dense(hiddenSize, hiddenSize, init='lecun_uniform'))
#model.add(Activation('linear'))
#model.add(Dropout(0.1))
model.add(Dense(hiddenSize, numOutputs, init='lecun_uniform'))
model.add(Activation('linear'))
model.load_weights("weights401")


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

#rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#model.compile(loss='mean_squared_error', optimizer=rms)

checkpointer = ModelCheckpoint(filepath="checkPt", verbose=1, save_best_only=True)
model.fit(X, Y, nb_epoch=1000, batch_size=20, verbose = 2, validation_split = 0.25, shuffle = True, callbacks=[checkpointer])
y_pred = model.predict(Xtest, batch_size=10000)
errors = Ytest-y_pred
print(errors)
print(np.mean(np.sum(abs(errors), axis = 1))) #Show the average total error
plt.hist(reject_outliers(np.sum(abs(errors), axis = 1)),100)
model.save_weights("weights402")
# Saving the objects:
with open('objs.pickle', 'w') as f:
    pickle.dump([obj0, obj1, obj2], f)

#deNormalizedErrors = Normalizers.deGauss(trainingOutputs,(Ytest-y_pred))
#print(deNormalizedErrors)
#print(mean(sum(abs(deNormalizedErrors), axis = 1)))

