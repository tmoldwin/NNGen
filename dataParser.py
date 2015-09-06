# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:30:21 2015

@author: tmold_000
"""
import numpy as np


def parse(fname):
    numInputs= 31;
    numOutputs = 28;
    numCols = numInputs + numOutputs + 1;
    generations = [];
    generationsToInputs = {}
    generationsToOutputs = {}
    with open(fname) as f:
        content = f.readlines()
    i = 0;    
    while i < len(content):
        strSplt = content[i].split()
        if strSplt[0] == 'Generation':
            generation = int(strSplt[1])
            generations.append(generation)
            i = i + 1;
            strSplt = content[i].split()
            outputs = [[]]
            inputs = [[]]
            while(len(strSplt) > 3):
                if len(strSplt) != 60:
                    print(len(strSplt))
                    print(i)
                outputVec = [float(x) for x in strSplt[1:numOutputs+1]]
                
                inputVec = [float(x) for x in strSplt[numOutputs+1:numCols+1]]
                outputs.append(outputVec)
                inputs.append(inputVec)
                i = i + 1;
                strSplt = content[i].split()

            generationsToInputs[generation] = inputs
            generationsToOutputs[generation] = outputs
        i = i+1
    return generations, generationsToInputs, generationsToOutputs
    
#Generate training sets and test sets from the data
def makeSets(X, Y, generations, proportionExamples, testProportion):
    In = []
    Out = []
    trainingInputs = []
    trainingOutputs = []
    testInputs = []
    testOutputs = []
    
    for i in range(len(generations)):
        outputVec = Y[generations[i]]
        inputVec = X[generations[i]]
        for n in range(len(inputVec)):
            inp = inputVec[n]
            outp = outputVec[n]
            #make sure that it's not empty and that it doesn't contain 250
            if(not(np.in1d(250, outp))  and len(outp) > 0): 
                In.append(inp)
                Out.append(outp)
#    while(In.count([])>0):
#        In.remove([])
#        Out.remove([])
    totalExamples = len(In)
    print(totalExamples)
    In = In[0:int(totalExamples*proportionExamples)]
    Out = Out[0:int(totalExamples*proportionExamples)]
    totalExamples = int(totalExamples*proportionExamples)
    print(totalExamples)
    inds = np.random.permutation(len(In))
    In = np.array([In[i] for i in inds])
    Out = np.array([Out[i] for i in inds])

    cutoffIndex = int(len(In)*(1-testProportion))
    trainingInputs = np.array(In[0:cutoffIndex]).astype(np.float32)
    trainingOutputs = np.array(Out[0:cutoffIndex]).astype(np.float32)
    testInputs = np.array(In[cutoffIndex:totalExamples]).astype(np.float32)
    testOutputs = np.array(Out[cutoffIndex:totalExamples]).astype(np.float32)
            
    return trainingInputs, trainingOutputs, testInputs, testOutputs

#generations, generationsToInputs, generationsToOutputs = parse(fname = "whole_population_0.txt")      
#trainingInputs, trainingOutputs, testInputs, testOutputs = makeSets(generationsToInputs, generationsToOutputs, generations[0:200], 0.5, 0.25)
