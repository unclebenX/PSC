# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 00:51:26 2017

@author: LIN Lu
"""


"""
Une module pour les cost_functions

"""

import RemoteAimsun as remote
import numpy as np
import matplotlib.mlab as mlab
from DataProcessing import featureNormalize1
from DataProcessing import featureNormalize
import csv





def surface(x,y):
    """ A artificial function for the test """
    #  mlab.bivariate_normal(sigma1, sigma2, x1, x2)
    Z1 = mlab.bivariate_normal(x, y, 1.5, 1.0, 2.0, 1.0)
    Z2 = mlab.bivariate_normal(x, y, 1.5, 0.5, -2, -1.5)
    Z3 = mlab.bivariate_normal(x, y, -2., -0.5, 1, 1)
    
    Z = 10.0 * (Z2 -Z1-Z3)
    return Z

def costFunction(X):
    """ A artificial function for the test """
    cost = surface(X[:,0], X[:, 1])
    N = len(X)
    return cost.reshape(N, 1)

def costFunction1(X):
    """ A artificial function for the test """
    cost = surface(X[0], X[1])
    return cost

def costFunction2(X):
    N = len(X)
    f = open("save_test.csv", "a")
    writer = csv.writer(f)
    
    cost = surface(X[:,0], X[:, 1])
    writer = csv.writer(f)
   
    for i in range(N):
        writer.writerow((X[i,0], X[i,1], cost[i]))
        
    f.close()
    return cost.reshape(N, 1)



#==============================================================================
#                      Old Aimsun commande  (A changer!!!)
#==============================================================================
def costFunctionAimsun(X):
    N = len(X)
    Y = np.zeros(N)
    aim = remote.Aimsun()            
    f = open("data.csv", "a")
    writer = csv.writer(f)
    
    for i in range(N):
        params1 = X[i][0]
        params2 = X[i][1]
        params = [params1, params2, 1., 1.] + [1. for i in range(101)] #refparams = [1.2, 0, 0, plein de trucs différents]
        aim.setParams(params)
        aim.simulate()
        loss_L2 = LossFunction2.GEHLoss("monty.sqlite", "monty_ref.sqlite")

        Y[i]=loss_L2
        writer.writerow((params1, params2, 1., 1., loss_L2))
#        print("Result", loss_L2)    
    f.close()
    aim.interrupt()
    return featureNormalize(Y)[0]

def costFunctionArtificiel(X):
    
    
    N = len(X)
    Y = np.zeros(N)
    D = Dataset("data_nuit_du_psc.csv")
            
    for i in range(N):    
        params1 = X[i][0]
        params2 = X[i][1]
        params = [params1,1.,params2]
        loss = D.value(params)
        Y[i]=loss
        
    return featureNormalize(Y)[0]


def costFunctionArtificiel2(X):
    
    
    N = len(X)
    Y = np.zeros(N)
    D = Dataset("data_nuit_du_psc.csv")
            
    for i in range(N):    
        params1 = X[i][0]
        params2 = X[i][1]
        params = [params1,1.,params2]
        loss = D.value(params)
        Y[i]=loss
        
    return Y

def initiation():
    D = Dataset("data_two_parameters.csv")
    D.bivariateSplineInit(D.INDEX_L1) 


def SimulationAimsun(X): 
    N = len(X)
    f = open("save_test.csv", "a")
    writer = csv.writer(f)
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = lossFunction(X[i,:].reshape(1,2))
        writer.writerow((X[i,0], X[i,1], Y[i]))
        
    f.close()
#    print(Y.shape)
    return featureNormalize1(Y)

def SimulationAimsun_spsa(X):
    D = Dataset("data_two_parameters.csv") 
    Y = D.bivariateSpline(X,D.INDEX_L1)
    
#     Y = D.interpolateValue(X,D.INDEX_L1)    Très très long

    return Y


""" =========================== end of algorithm ================================ """












