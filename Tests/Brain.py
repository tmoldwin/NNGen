# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:36:03 2015

@author: tmold_000
"""
from pybrain.datasets import SupervisedDataSet
import numpy as np
import math
import dataParser
import Normalizers

numInputs = 31
numOutputs = 28

#generations, generationsToInputs, generationsToOutputs = dataParser.parse(fname = "whole_population_0.txt")

trainingInputs, trainingOutputs, testInputs, testOutputs = dataParser.makeSets(X = generationsToInputs, Y = generationsToOutputs, generations = generations[130:200], examplesPerGeneration = 500, testProportion = 0.25)
#
#X = Normalizers.gaussNormalize(trainingInputs)
#Xtest = Normalizers.gaussNormalize(testInputs)
#
#Y = Normalizers.gaussNormalize(trainingOutputs)
#Ytest = Normalizers.gaussNormalize(testOutputs)

X = trainingInputs
Xtest = testInputs
Y = trainingOutputs
Ytest = testOutputs

trainingSet = SupervisedDataSet(numInputs, numOutputs)
for x, y in zip(X, Y):
    trainingSet.appendLinked(x, y)

#----------
# build the network
#----------
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(numInputs,
                   25, # number of hidden units
                   numOutputs,
                   bias = True,
                   hiddenclass = SigmoidLayer,
                   outclass = LinearLayer
                   )
#----------
# train
#----------
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, trainingSet, verbose = True)
trainer.trainUntilConvergence(maxEpochs = 100)

#----------
# evaluate
#----------
import pylab
# neural net approximation
pylab.plot(Xtest,
           [ net.activate([x]) for x in Xtest ], linewidth = 2,
           color = 'blue', label = 'NN output')

# target function
pylab.plot(Ytest,
           yvalues, linewidth = 2, color = 'red', label = 'target')

pylab.grid()
pylab.legend()
pylab.show()

