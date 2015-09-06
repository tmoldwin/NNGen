# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 13:49:05 2015

@author: toviah.moldwin
"""

import pickle
import Normalizers
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import scipy.stats as st

def dePickle(f):
    f = open(f)
    trainingInputs, trainingOutputs, testInputs, testOutputs, X, Y, Xtest, Ytest, y_pred = pickle.load(f)
    f.close()
    return trainingInputs, trainingOutputs, testInputs, testOutputs, X, Y, Xtest, Ytest, y_pred

    
def sumAndSort(X):
    sm = np.sum(abs(X), axis = 1)
    indices = sm.argsort()
    ranking = st.rankdata(sm)
    return ranking, sm, X[indices]
    
def compareRankings(rnk1, rnk2):
    difs = np.abs(np.subtract(rnk1, rnk2))
    return difs, np.mean(difs)
    

#f = open('store.pckl', 'w')
#pickle.dump([trainingInputs, trainingOutputs, testInputs, testOutputs, X, Y, Xtest, Ytest, y_pred], f)
#f.close()

##Test
#A = np.array([[1, 2, 3], [10, 11, 12], [5,6,-7]])
#B = np.array([[10,11,12], [5,6,-7], [1,2,3]])
#inds1, sm1, A1 = sumAndSort(A)
#inds2, sm2, B1 = sumAndSort(B)
#print(inds1, sm1, A1)
#print(inds2, sm2, B1)
#difs, mn = compareRankings(inds1, inds2)
#print(difs, mn)



#trainingInputs, trainingOutputs, testInputs, testOutputs, X, Y, Xtest, Ytest, y_pred = dePickle('store.pckl');
#normalizedY = Normalizers.deGauss(trainingOutputs, Ytest)
#mormalizedPred = Normalizers.deGauss(trainingOutputs, y_pred)

rnk1, sm1, ySort = sumAndSort(Ytest)
rnk2, sm2, predSort = sumAndSort(y_pred)
difs, mn = compareRankings(rnk1, rnk2)
print(mn)
plt.hist(difs,100)

#inds3 = np.random.permutation(len(Ytest))
inds3 = range(len(Ytest))
difs2, mn2 = compareRankings(inds1, inds3)
plt.hist(difs2,100)
print(mn2)



#errors = normalizedY - mormalizedPred 
#print(np.mean(abs(errors),axis = 0));
#avgError = np.mean(np.sum(abs(errors), axis = 1))
#print(avgError) #Show the average total error


