# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:23:56 2015

@author: tmold_000
"""
import numpy as np

def gaussNormalize(x):
    dev = x.std(axis=0)
    dev[(dev == 0) ] = 1
    xMean = np.tile(x.mean(axis=0), (len(x),1))
    devTiled = np.tile(dev, (len(x),1))
    normalizedX = (x - xMean) / devTiled;
    return normalizedX
    
def minMaxNormalize(x):
    dif = x.max(axis=0) - x.min(axis=0)
    dif[(dif == 0) ] = 1
    xMin = np.tile(x.min(axis=0), (len(x),1))
    difTiled = np.tile(dif, (len(x),1))
    normalizedX = (x - xMin) / difTiled
    return normalizedX
    
def deGauss(x, xNorm):
    xMean = np.tile(x.mean(axis=0), (len(xNorm),1))
    xStd = np.tile(x.std(axis=0), (len(xNorm),1))
    return (xNorm *xStd) + xMean
    
    
def deMinMax(x, xNorm):
    xMin = np.tile(x.min(axis=0), (len(xNorm),1))
    xMax = np.tile(x.max(axis=0), (len(xNorm),1))
    return (xNorm*(xMax-xMin))+xMin
    

def reject_outliers(data, m = 2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
#    
#A = np.array([[1, 5.7, 1], [1, 10.4, 2], [1, 9, 3]]);
#print(A)
#gN = gaussNormalize(A)
#print("Gauss Normalize")
#print(gN)
#print("deGuass")
#print(deGauss(A, gN))
#
#mmN = minMaxNormalize(A)
#print("mmN Normalize")
#print(mmN)
#print("deMM")
#print(deMinMax(A, mmN))