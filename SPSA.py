# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:56:43 2017

@author: LIN Lu, MEMBRADO Jean-Baptiste
"""


import numpy as np
from LossFunctions import *
import pickle
import glob
import os.path as op


#==============================================================================
#                        Les Hyperparamètres
#==============================================================================
#==============================================================================
SAVE_PARAMS_EVERY = 100
#==============================================================================
iteration = 1000

# Paramètres utilisés pour les structures classiques des suites (ak) et (ck)

a = 1.0  # >= 1.0 car Rademacher distribution X < a     
c = 0.5
gamma = 2/6  # gamma doit être compris dans [1/6,1/2]


entier = np.arange(1, iteration+1)
sa = a / entier
sc = c / (entier **gamma)
#==============================================================================




def load_saved_params():
    """
    A helper function.
    Cette fonction nous permet de reprendre SPSA au point où on s'est arrêté à cause 
    d'une panne. Cela évite de re-calculer tous les calculs.
    
    """
    st = 0
    for f in glob.glob("saved_gradient_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("saved_gradient_{}.npy".format(st), "rb") as f:
            params = pickle.load(f)
        return st, params
    else:
        return st, None


def save_params(iter, params):
    """ 
    A helper function.
    Cette fonction nous permet d'enregistrer les données pour que ce programme soit
    tolérant aux pannes.
    
    iter : Pour chaque SAVE_PARAMS_EVERY itération, on enregistre la position.
    
    """
    with open("saved_gradient_{}.npy".format(iter), "wb") as f:
        pickle.dump(params, f)


""" 
# Une version raccourcie du SPSA (sans dessin, dimension >= 3) 


def spsa(f,x, iteration, s_a, s_c):
    dim = len(x)

        

    for i in range(iteration):

        delta = 2*np.random.binomial(1, 0.5, dim)-1.
        fp = f(x+s_c[i]*delta)
        fn = f(x-s_c[i]*delta)                            
        grad = (fp-fn)/(2*s_c[i]*delta)
        cost = (fp+fn) / 2.
        x = x- s_a[i]*grad
                     
    return (x, cost)
    
"""


def spsa(f,x, iteration, s_a, s_c, useSaved = False, PRINT_EVERY=1000):
    """
    f : La fonction étudiée à minimiser
    x : Le vecteur de départ choisi
    iteration : Nombre d'itérations que l'on se fixe
    s_a : suite (ak) sous la forme np.array
    c_a : suite (ck) sous la forme np.array
    useSaved : si mis à True, permet de reprendre le point où l'on s'est arrêté
    PRINT_EVERY = n : Détermine la fréquence à laquelle on affiche le vecteur en cours
    """
    dim = len(x)
    
    if useSaved:
        start_iter, oldx = load_saved_params()
        if start_iter > 0:
            x = oldx;

    else:
        start_iter = 0
    
    
    parcours = np.zeros((iteration+1)*dim).reshape(iteration+1,dim)
    parcours[0] = x
    
    cost = -1.
    for i in range(start_iter, iteration):

        
        #Distribution aléatoire classique pour les vecteurs aléatoires Delta k
        #Distribution de Rademacher
        delta = 2*np.random.binomial(1, 0.5, dim)-1.
                                    
        fp = f((x+s_c[i]*delta).reshape(-1,dim))
        fn = f((x-s_c[i]*delta).reshape(-1,dim))   

        #Calcul du gradient estimé           
        grad = (fp-fn)/(2*s_c[i]*delta)
        
        cost = (fp+fn) / 2.
               
        # Mise à jour du vecteur x        
        x = x - s_a[i]*grad
                            
        if i % PRINT_EVERY == 0 and i!=0:
            print("Iteration {0}: {1}".format(i, cost))

        if i % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(i, x)
            
        #Stockage du parcours
        parcours[i+1] = x

    return (x, parcours, cost)


""" =========================== end of algorithm ================================ """







