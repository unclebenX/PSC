# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:14:15 2017

@author: LIN Lu
"""


"""
 Main algorithm.
 
"""


import numpy as np


import matplotlib.pyplot as plt
from sklearn.svm import SVR

from sklearn.externals import joblib
from  LossFunctions import *
from DataProcessing import *
from DataProcessing import featureNormalize1
from GeneticFilter import filtre_barycentre

from time import time as t



#==============================================================================
#                          Paramètres à saisir
#==============================================================================

#==============================================================================
# Filtre : filtre_barycentre
#==============================================================================
n_tirage_bary = 50000    # Le nombre de points de tirage
#==============================================================================




def sigmoid(z):
    g = 1. / (1 + np.exp(-z))
    
    return g


def filtre(X, Y, seuil):
    """
    A simple filter function using Monte-Carlo
    """
    Z = np.less(Y, seuil)  # Z.shape =  N * 1  Z type = boolean
    Y_bis = np.extract(Z, Y)
    R = X[Z]
    n = sum(Z)
    print("Il reste " + str(n) + " de donnees apres le filtrage.")
    return (R, Y_bis, n)


def check(X, seuil):
    """
    A helper function that check the distribution of postive sample/ negative sample
    """
    N = len(X)
    if N == 0:
        return (0,0)
    else:
        x = featureNormalize1(X[:,-1])
        x = sigmoid(x)
        x = np.greater(x, seuil)
        pourcentre = 1.- np.sum(x)/N
        return (pourcentre, N)





def Genetique_svm(f_loss,N, centre,forcer = False, m = 2, taille = 3., C = 1e2, \
                  gamma='auto', seuil = 0.502, num_cluster = 3, fileName = "save_test" ):
    """
    0 iteration of "Genetic Algorithm"
    f_loss : study function
    centre : center of carte
    forcer : For first application, we need forcer = True. After first train, we can 
            set forcer = False
    m : dimension of manifold = nomber of features
    taille : could be a float (scalaire) or a np.array
    num_cluster : Nomber of "valley" estimated
    fileName : Save data historic with fileName
    
    return : Bon points de départ pour SPSA
    """
    
    # Défaut paramètre: "rayon = 3.0 /N"
    rayon = 3.0/N   # Attention il faut modifier cela pour dimension >2
        
    oldx = np.loadtxt(open(fileName+".csv", "rb"), delimiter=",", skiprows=0)  ### 小心 shape!

    pourcentage0, oldN = check(oldx, seuil)
    print(pourcentage0)
    while(pourcentage0<0.3 or pourcentage0>0.8 or forcer):
        
        params = centre + np.zeros(m).reshape(1,-1)  # params.shape = 1 * m
        a = (2 * np.random.rand(N, m)- 1.) *taille
        #    A = A.reshape(N, 2)
        x = a + params    # x est de taille N * m ;
        
        loss = f_loss(x)  # loss.shape = N * 1
        
        
        y = loss
        # Define parameters
        y = np.array(y).reshape(N, 1)
        y = sigmoid(y)
        y = np.greater(y, seuil)
        pourcentage0 = ((N- np.sum(y))+ pourcentage0*oldN)/(N+oldN)
        oldN += N
        # positive = ce qu'on veut i.e. La disctance ~= 0
#==============================================================================
#         print("================================================================")
#         print("Pourcentage d'echantillons positifs : ", pourcentage0)
#         print("================================================================")
#==============================================================================
        if forcer:
            forcer = False
        

#==============================================================================
#==============================================================================
#             更改 dimesions 時要注意底下這幾行!!!!!!!
#==============================================================================
#==============================================================================
    A = np.loadtxt(open(fileName+".csv", "rb"), delimiter=",", skiprows=0)
    X = A[:,:m]   ### Attension aux dimensions !!!!!
    lenth = len(X)

    Y = A[:,-1]

    Y = np.array(Y).reshape(lenth, 1)
    Y = featureNormalize1(Y.ravel())
    Y = sigmoid(Y)
    Y = np.greater(Y, seuil)
    Y = Y.astype(float)
      

    #Define and train machine
    clf = SVR(kernel ='rbf', C = C , gamma = gamma)
    clf.fit(X, Y.ravel())
    joblib.dump(clf,"./algo_genetique_0.pkl")
    

    
    # Affiche les dessins
    plt.figure(0)
    plt.scatter(X[:, 0], X[:, 1], c = Y,s = 40, facecolors='none', zorder=10, cmap=plt.cm.bwr)
    
    plt.axis('tight')
    x_min, y_min = (centre - taille)[0,0], (centre - taille)[0,1]  #X[:, 0].min()
    x_max, y_max = (centre + taille)[0,0], (centre + taille)[0,1]     #X[:, 0].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    
    clf = joblib.load("./algo_genetique_0.pkl")
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)

    plt.pcolormesh(XX, YY, Z)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    st = t()
    bary = filtre_barycentre(clf.decision_function, n_tirage_bary, taille, centre, rayon, num_cluster = num_cluster)
    et = t()
    print("Temps de calcul des barycentres = " + str(et-st))
    
    plt.scatter(bary[:, 0], bary[:, 1], c = 'black', s = 200)

    plt.title('Resultat apres 0 iterations')

    plt.savefig('Iteration_0.png')
    plt.show()
    return bary



def Genetique_svm_iter(f_loss, N, centre,m = 2, taille = 0.5, C = 1e3, gamma=0.1,\
                       seuil_filtre = 0.5, seuil = 0., i = 1, seuil3 = 0.502, num_cluster = 3):
    """
    0 iteration of "Genetic Algorithm"
    f_loss : study function
    centre : center of carte
    forcer : For first application, we need forcer = True. After first train, we can 
            set forcer = False
    m : dimension of manifold = nomber of features
    taille : could be a float (scalaire) or a np.array
    i : index of iteration
    num_cluster : Nomber of "valley" estimated
    fileName : Save data historic with fileName
    
    
    """
    rayon = 3./N
    
    params = centre + np.zeros(m).reshape(1,-1) # params.shape = 1 * m
    A = taille * (2 * np.random.rand(N, m)- 1.) #    A = A.reshape(N, 2)
    X = A + params               # X est de taille N * m ;

#==============================================================================
#   Partie pour le filtrage
#==============================================================================
    clf = joblib.load("./algo_genetique_" + str(i-1)+".pkl")
    g = clf.decision_function
    loss = clf.decision_function(X)  # loss.shape = N * 1
    Y = loss

    X, Y, oldn = filtre(X, Y.ravel(), seuil_filtre)
    #    Y = sigmoid(Y)
    # 如果沒有使用 Y = sigmoid(Y) 那 seuil 可以改成 0
    Y = np.greater(Y, seuil)
    Y = Y.astype(float)
    pourcentage0 = 1.- np.sum(Y)/oldn
    print("Pourcentage d'echantillons positifs : ", pourcentage0, "==============================")
    #    assert 0.7 > pourcentage0 > 0.3
    
    # ================================================
    # positive = ce qu'on veut i.e. La disctance ~= 0
    # ================================================
    k = 2
    while((pourcentage0<0.3 or pourcentage0>0.7) and k <5):
        params = centre + np.zeros(m).reshape(1,-1)
        A = taille * (2 * np.random.rand(k*N, m)- 1.)
        X = A + params
        loss = g(X)
        X, Y, n = filtre(X, loss.ravel(), seuil_filtre)
        Y = np.greater(Y, seuil)
        pourcentage0 = ((n- np.sum(Y)) + pourcentage0*oldn)/(n+oldn)
        print("Il reste au total " + str(n+oldn)+ " donées")
        print("================================================================")
        print("Pourcentage d'echantillons positifs : ", pourcentage0, "==============================")
        print("================================================================")
        k += 1
    

    
    loss = f_loss(X)
    Y = loss
    lenth = len(Y)
    
    #Define parameters
    Y = np.array(Y).reshape(lenth, 1)
    Y = sigmoid(Y)
    Y = np.greater(Y, seuil3)
    
    #Define and train machine
    clf = SVR(kernel ='rbf', C = C , gamma = gamma)
    clf.fit(X, Y.ravel())
    joblib.dump(clf,"./algo_genetique_" + str(i)+".pkl")
    
    plt.figure(i)
    
    # Affiche les dessins
    plt.scatter(X[:, 0], X[:, 1], c = Y,s = 40, facecolors='none', zorder=10, cmap=plt.cm.bwr, alpha = 0.5)
    
    x_min, y_min = (centre - taille)[0,0], (centre - taille)[0,1]  #X[:, 0].min()
    x_max, y_max = (centre + taille)[0,0], (centre + taille)[0,1]     #X[:, 0].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    
    clf = joblib.load("./algo_genetique_" + str(i)+".pkl")
    f = clf.decision_function 
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])


    st = t()
    bary = filtre_barycentre(f, n_tirage_bary , taille, centre, rayon, num_cluster = num_cluster)  #n_ech_bary rayon
    et = t()
    print("Temps de calcul des barycentres = " + str(et-st))

    plt.scatter(bary[:, 0], bary[:, 1], c = 'black', s = 200)


    plt.title('Resultat apres '+ str(i) + ' iterations')
    plt.savefig('Iteration_'+str(i)+'.png')
    plt.show()
    return bary
    
""" =========================== end of algorithm ================================ """



"""
Some tests of developer :
    

def algo_genetique(f,N, n,centre, taille = 0.5, C =1., gamma=0.09): # N = le nombre d'échantillons, n = le nombre d'itérations
    Genetique_svm(f,N, centre, C =C, taille =taille, seuil = 0.502, gamma=gamma)  # seuil = 0.502
    
    # seuil = temperature  seuil should be 0. ~ 0.9
    # Conseil d'utilisation seuil = 0.05

    for j in range(n):
        Genetique_svm_iter(N, centre, i = j + 1,C =1., taille =taille, gamma=0.8, seuil_filtre = 0.50 ,seuil = 0.0)
    # seuil_filtre 可以用來調控 échantillon positives 的比例: seuil_filtre 越小 
    # échantillon positives 的比例就越高
    
    # échantillons positives < Seuil < échantillons négatives < Seuil_filtre    
    
    # 如果 第一個 SVM 沒有 overfit (不夠精細 ex: 取了 marge!)，那就必須放寬 seuil_filtre 以免誤殺。
    # 必須放寬(提高) seuil 以免誤殺。


def algo_genetique2(N, n): # N = le nombre d'échantillons, n = le nombre d'itérations
    Genetique_svm(N,C =10, seuil = 0.5)  # seuil = 0.502
    
    # seuil = temperature  seuil should be 0. ~ 0.9
    # Conseil d'utilisation seuil = 0.05

    for j in range(n):
        Genetique_svm_iter(N, i = j + 1,C =0.1, seuil_filtre = 0.5 + 0.01*j ,seuil = 0.02)
    # seuil_filtre 可以用來調控 échantillon positives 的比例: seuil_filtre 越小 
    # échantillon positives 的比例就越高
    
    # échantillons positives < Seuil < échantillons négatives < Seuil_filtre    
    
    # 如果 第一個 SVM 沒有 overfit (不夠精細 ex: 取了 marge!)，那就必須放寬 seuil_filtre 以免誤殺。
    # 必須放寬(提高) seuil 以免誤殺。
     
def contour(n):
    XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
    for j in range(n+1):
        clf = joblib.load("algo_genetique_" + str(j)+".pkl")
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.figure(j + n +1)
        plt.pcolormesh(XX, YY, Z>0 , cmap=plt.cm.Paired)
        plt.title('Le resultat apres '+ str(j) + ' iterations')
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
        plt.savefig('Contour_Iteration_'+str(j)+'.png')

#Genetique_svm(100, m = 2, taille = 4., C = 1e1, gamma=0.05, seuil = 0.5)

#Centre = np.ones(2).reshape(1,-1)

#algo_genetique(costFunctionArtificiel, 200, 1,Centre, taille = 0.5, C =1., gamma=0.1)  #0.097



#==============================================================================
# Paramètre réussis :
#  C = 1., gamma=0.09, seuil = 0.502  pour Genetique_svm
#  C = 0.1, gamma=0.8, seuil_filtre = 0.50 ,seuil = 0.0 pour Genetique_svm_iter
#==============================================================================

"""









