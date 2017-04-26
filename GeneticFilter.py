# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:27:31 2017

@author: LIN Lu
"""

"""
Une module pour filtrer les points.

"""


import numpy as np


def filtre1(f, X, seuil = 0.502):
    """
    First filter : Monte-Carlo, puis garder les points qui sont dans la zone d'intérêt
    f : Fonction étudiée
    X : Une matrice des points  X.shape = N * m, N = le nombre de points, m = dimension
    
    """
    loss = f(X).ravel()  


    Z = np.less(loss, seuil) # Y.shape =  N * 1  Y type = boolean  

    R = X[Z]   # Filtrer les points
    n = len(R) # = sum(Z) 
    print("Il reste " + str(n) + " de données après le premier filtrage.")
    return R



def centre_loc(X, rayon, num_cluster = 3) :
    """
    A function that return num_cluster points of m-dimension space R^m
    L'entrée X: a matrix of point after filter1
    rayon : Le rayon de la boule
    """
    m = X.shape[1]
    a = np.zeros(num_cluster* m).reshape(num_cluster, m)
    c = np.zeros(num_cluster* m).reshape(num_cluster, m)
    num = np.zeros(num_cluster)
    

    
    for i in np.arange(num_cluster):
        a[i] = X[i]

        distance = np.sum((X - a[i])**2, axis = 1)

        Z = np.less(distance, rayon **2)

        R = X[Z]
        num[i] = len(R)
        print("Pour "+str(i)+"ème centre, la taille du cluster est " +str(num[i]))
        c[i] = sum(R) / num[i]
    return c



def filtre_barycentre(f, N, taille, centre, rayon, seuil = 0.502, num_cluster = 3, m = 2):
    params = centre + np.zeros(m).reshape(1,-1) # params.shape = 1 * m
    A = taille * (2 * np.random.rand(N, m)- 1.) #    A.shape = (N, m)  m = dimension
    X = A + params   # Une matrice de shape N * m

    X_1 = filtre1(f, X, seuil= seuil)

    return centre_loc(X_1, rayon, num_cluster = num_cluster)


""" =========================== end of algorithm ================================ """










