# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:29:27 2017

@author: LIN Lu
"""

"""
Une module qui regroupe les différentes méthodes de normalisation. Etant contraint
par le temps, nous n'avons pas pu implémenter les autres méthodes plus avancée 
comme PCA 

"""


import numpy as np
#from sklearn import preprocessing

def normalizeRows(x):
    """ Row normalization function """
    y = np.linalg.norm(x,axis=1,keepdims=True)
    x /= y

    return x


#==============================================================================
#                           Standardization
#           Cela égale à  X_scaled = preprocessing.scale(X)
#==============================================================================

# Normaliser les "Features" (caractéristiques)  -----DIMENSION n
def featureNormalize(X):
    x_mean = X.mean(axis = 0) # Moyenne
    sigma = X.std(axis = 0)   # Ecart type
    x_norm = (X - x_mean) / sigma
    return (x_norm, x_mean, sigma)

def featureNormalize1(X):
    n = len(X)
    if n == 1:
        return X
    else:
        x_mean = X.mean(axis = 0) # Moyenne
        sigma = X.std(axis = 0)   # Ecart type
        x_norm = (X - x_mean) / sigma         
    return x_norm


#==============================================================================
#                       4.3. Preprocessing data
#          Standardization, or mean removal and variance scaling
#==============================================================================

#    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing