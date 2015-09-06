# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 11:19:47 2015

@author: tmold_000
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from pandas import *
import parser

def reject_outliers(data, m = 2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
#generations, generationsToInputs, generationsToOutputs = parse(fname = "whole_population_0.txt")
numOutputs = 28;


trainingInputs, trainingOutputs, testInputs, testOutputs = makeSets(generationsToInputs,  generationsToOutputs, generations[50:200],500, 0)


trainingInputs = array(trainingInputs).astype(float32)
trainingOutputs = array(trainingOutputs).astype(float32)
testInputs = array(testInputs).astype(float32)
testOutputs = array(testOutputs).astype(float32)

for i in range(numOutputs):
    data = trainingOutputs[:,i]
    data = reject_outliers(data)
    subplot(4, 7,i)
    hist(trainingOutputs[generations[i]],10)