# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:03:52 2015

@author: tmold_000
runfile('/home/ls/users/toviah.moldwin/Dropbox/IdanLab/PythonFiles/NNGen/Lasagne.py', wdir='/home/ls/users/toviah.moldwin/Dropbox/IdanLab/PythonFiles/NNGen')
"""
from __future__ import print_function
import numpy as np
import Normalizers
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import lasagne
import dataParser
import matplotlib.pyplot as plt
import time
import pickle


numInputs = 31;
numOutputs = 28;
def createNet(X, Y, ln, loadFile = ""):
    net1 = NeuralNet(
        layers=[  # four layers: two hidden layers
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('hidden1', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters: Best 400 400
        input_shape=(None, numInputs),  # 31 inputs
        hidden_num_units=400,  # number of units in hidden layer
        hidden1_num_units=400,
        hidden2_num_units=400,
        hidden3_num_units=400,
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=numOutputs,  # 4 outputs
    
        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=ln,
        update_momentum=0.9,
    
        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=1500,  # we want to train this many epochs
        verbose=1,
        )
    #if (loadFile != ""):
        #net1.load_params_from(loadFile)
    net1.max_epochs = 50
    net1.update_learning_rate = ln;

    return net1


    
generations, generationsToInputs, generationsToOutputs = dataParser.parse(fname = "whole_population_0.txt")
iters = 150
saveFile = "LasagneWeights400_2Layer"
trainingInputs, trainingOutputs, testInputs, testOutputs = dataParser.makeSets(generationsToInputs, generationsToOutputs, generations[0:200], 1, 0.25)
ln = 0.01
X = Normalizers.gaussNormalize(trainingInputs)
Xtest = Normalizers.gaussNormalize(testInputs)
#
Y = Normalizers.gaussNormalize(trainingOutputs)
Ytest = Normalizers.gaussNormalize(testOutputs)

X = np.asarray(X, np.float32)
Y = np.asarray(Y, np.float32)
Xtest= np.asarray(Xtest, np.float32)
Ytest = np.asarray(Ytest, np.float32)
net = createNet(X, Y, ln, saveFile)
for n in range(iters): 
    net.fit(X, Y) # This thing try to do the fit itself
    y_pred = net.predict(Xtest)
    errors = Normalizers.deGauss(trainingOutputs, Ytest) - Normalizers.deGauss(trainingOutputs, y_pred)
    print(np.mean(abs(errors),axis = 0));
    avgError = np.mean(np.sum(abs(errors), axis = 1))
    print(avgError) #Show the average total error
    with open("avgs2", "a") as myfile:
        myfile.write(str(avgError)+"\n")
    plt.hist(Normalizers.reject_outliers(np.sum(abs(errors), axis = 1)),20)
    # Saving the objects:
with open('model.pickle', 'w') as f:
    pickle.dump(net, f)
#f = open('store.pckl', 'w')
#pickle.dump([trainingInputs, trainingOutputs, testInputs, testOutputs, X, Y, Xtest, Ytest, y_pred], f)
#f.close()
    plt.close()
    plt.title(avgError)
    plt.savefig("Two400")

    #net.save_params_to(saveFile)