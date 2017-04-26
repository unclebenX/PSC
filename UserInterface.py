# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:08:19 2017

@author: LIN Lu
"""
"""
使用者介面。
L'interface pour faciliter l'usage d'algorithme.
A user interface for simplify the use of algorithm.
Eine Benutzerschnittstelle.
"""

import numpy as np
import glob, os
import csv
from CSVProcessing import Dataset
from DataProcessing import featureNormalize1
from LossFunctions import *
from GeneticAlgorithm import Genetique_svm
from GeneticAlgorithm import Genetique_svm_iter
from GeneticFilter import filtre_barycentre
from SPSA import spsa


import matplotlib.pyplot as plt
from time import time as t







#==============================================================================
#                      Les Hyperparamètres SPSA
#==============================================================================
#==============================================================================
iteration = 100
a = 1.0       # make sure that a >= 1.0, since Rademacher distribution X < a     
c = 0.5
gamma = 2/6                         # gamma \in [1/6,1/2)


entier = np.arange(1, iteration+1)
sa = a / entier                     # Suite a_k
sc = c / (entier **gamma)           # Suite c_k
#==============================================================================




#==============================================================================
#                     Contrôle de GeneticAlgorithm
#==============================================================================
Forcer = True  # Set Forcer = False if you want to re-use old data

fileName = "Historique_de_"+"INDEX_L2" # Savet data in file "fileName.csv" à changer

#==============================================================================
#                   Autres initialisations (for interpolation test)
#==============================================================================
#D = Dataset("data_two_parameters.csv") 
#lossFunction = D.lossFunction(D.INDEX_L1)
#D = Dataset("data_two_parameters_reactiontime_reactiontimeattrafficlight.csv") 
#lossFunction = D.lossFunction(D.INDEX_L2)
#==============================================================================

""" ===================== Nettoyage du répertoire ========================  """
"""  
清理資料夾，讓舊的資料不會引想到新的訓練結果。
Clean Directory such that old data training result make no influences to new result.

"""
#

def creatFile(fileName="save_test1"):
    f = open(fileName+".csv", "a")
#    writer = csv.writer(f)
    f.close()

def clearCSV(fileName = "save_test"):
    f = open(fileName+".csv", "w")
    writer = csv.writer(f)
    f.close()

def clearPKL(k):
    f = open("algo_genetique_"+str(k)+".pkl", "w")
    writer = csv.writer(f)
    f.close()

def removePKL(k):
    os.remove("algo_genetique_"+str(k)+".pkl")

def removePKLrecursive():
    filelist = glob.glob("*.pkl")
    for f in filelist:
        os.remove(f)

""" ===================== Nettoyage du répertoire ========================  """


""" ===================== Interpolation Function =============================  """
#=====================================================================
# A cause d'initiation de la classe, on est obligé de les définir ici.
# On n'a pas encore une solution pour contourner ce problème.
#=====================================================================
def SimulationAimsun0(X): 
    Y = lossFunction(X.reshape[:,2])
    return featureNormalize1(Y)

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

def SimulationAimsun1(X,fileName = fileName): 
    N = len(X)
    f = open(fileName+".csv", "a")
    writer = csv.writer(f)
    Y = np.zeros(N)

    for i in range(N):
#            print(lossFunction(X[i,:].reshape(1,2)))
        Y[i] = lossFunction(X[i,:].reshape(1,2))
        writer.writerow((X[i,0], X[i,1], Y[i]))

    f.close()

    return featureNormalize1(Y)
""" =================== End of Interpolation Function ==========================  """



def main(f, N, n, iteration,centre, taille, C =1., gamma=0.1, num_cluster = 3, fileName = "save_test"): # N = le nombre d'échantillons, n = le nombre d'itérations
    """
    f : study function
    N : Nomber of points of first Monte-Carlo
    n : Nomber of iteration of "Genetic-Algorithm"
    centre : center of map
    taille : could be a float (scalaire) or a np.array. Size of map
    num_cluster : Nomber of "valley" estimated
    fileName : Save data historic with fileName
    void function
    """
    depart = Genetique_svm(f, N, centre, forcer = Forcer, C =C, taille =taille, seuil = 0.502, \
                           fileName = fileName, gamma=gamma, num_cluster= num_cluster)  # seuil = 0.502
    print("==============================================================")
    print("Après 0 itération les points de départ trouvés sont "+str(depart))
    print("==============================================================")
    # seuil = temperature  seuil should be 0. ~ 0.9
    # Conseil d'utilisation seuil = 0.05
    for j in range(n):
        depart = Genetique_svm_iter(f, N, centre, i = j + 1,C =1., taille =taille, gamma=0.8, seuil_filtre = 0.50 ,seuil = 0.0, num_cluster= num_cluster)
        print("==============================================================")
        print("Après " +str(j+1)+" itération les points de départ trouvés sont "+str(depart))
        print("==============================================================")
#    print("La solution 1 du genetique est :" )
#    print("\t Reaction time : ", D.denormalizeValue(depart[0][0], 0))
#    print("\t Lookahead Distance : ", D.denormalizeValue(depart[0][1], 1))
#    
#    print("La solution 2 du genetique est :" )
#    print("\t Reaction time : ",  D.denormalizeValue(depart[1][0], 0))
#    print("\t Lookahead Distance : ",  D.denormalizeValue(depart[1][0], 1))
    st = t()
    x0, parcours0, cost0= spsa(f, depart[0], iteration,sa,sc)
    x1, parcours1, cost1= spsa(f, depart[1], iteration,sa,sc)
    et = t()
    print("Temps de calculs = " + str(et-st))
    print("La solution du parcours0"+str(depart[0])+" est " +str(x0))
    print("La solution du parcours1"+str(depart[1])+" est " +str(x1))
    
    
#    print("La solution du parcours0 est :" )
#    print("\t Reaction time : ", D.denormalizeValue(x0[0], 0))
#    print("\t Lookahead Distance : ", D.denormalizeValue(x0[1], 1))
#    print("La valeur de perte associée est : ", cost0)#f(np.array([x0]))[0])
#    print("La solution du parcours1 est :" )
#    print("\t Reaction time : ",  D.denormalizeValue(x1[0], 0))
#    print("\t Lookahead Distance : ",  D.denormalizeValue(x1[1], 1))
#    print("La valeur de perte associée est : ", cost1)#f(np.array([x1]))[0])
    
    
    
    x_min, y_min = (centre - taille)[0,0], (centre - taille)[0,1]  #X[:, 0].min()
    x_max, y_max = (centre + taille)[0,0], (centre + taille)[0,1]     #X[:, 0].max()

    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    plt.scatter(parcours0[0,0], parcours0[0,1], c = 'black', s = 200)
    plt.scatter(parcours1[0,0], parcours1[0,1], c = 'r', s = 200)
    plt.scatter(parcours0[:,0], parcours0[:,1], c = 'black', s =40)
    plt.scatter(parcours1[:,0], parcours1[:,1], c = 'r', s =40)
    plt.plot(parcours0[:,0],parcours0[:,1])
    plt.plot(parcours1[:,0],parcours1[:,1])
    plt.title('SPSA avec ' + str(iteration)+' iteration')
    plt.savefig("Illustration d'algorithme SPSA "+ str(iteration)+".png")
    plt.show()
    
    Data_save = np.loadtxt(open(fileName+".csv", "rb"), delimiter=",", skiprows=0)
    
    print("==============================================================")
    print("Le nombre d'appels Aimsum par SPSA est..." +str(2*iteration))
    print("==============================================================")
    print("Le nombre d'appels Aimsum au total est..." +str(len(Data_save)))
    print("==============================================================")

#==============================================================================
#                              User commande
#==============================================================================
removePKLrecursive()
clearCSV(fileName = fileName)
creatFile(fileName = fileName)
Centre = np.array([0.5, 0.5]).reshape(1,2)
taille = 0.5 # taille = np.array([0.5, 0.5])
main(SimulationAimsun1, 300, 1, iteration, Centre, taille, gamma = 'auto', C = 10., fileName = fileName)

#==============================================================================


  


#==============================================================================
#                          Tests du développeur
#==============================================================================
def spas_test(f,x0,PRINT_EVERY=50):
    """=================== Visulize SPSA gradient descent ====================== """
    # x0 = np.array([-1.5, 1.])
    st = t()
    x1, parcours1, cost1 = spsa(f, x0, iteration,sa,sc,PRINT_EVERY=PRINT_EVERY)
    et = t()
    
    print("La solution du parcours0 est " +str(x1))
    print("Temps de calculs = " + str(et-st))
    plt.scatter(parcours1[0,0], parcours1[0,1], c = 'r', s =200)
    plt.scatter(parcours1[:,0], parcours1[:,1], c = 'r')

    plt.plot(parcours1[:,0],parcours1[:,1])
    plt.title('SPSA avec ' + str(iteration)+' iteration')
    plt.savefig("Illustration d'algorithme SPSA "+ str(iteration)+".png")
    plt.show()

#spas_test()
#==============================================================================



#==============================================================================
#                          Test pour l'analyse 
#==============================================================================

"""
@author: LIN Lu, GUO Louis, BROUX Lucas

"""


"""====================Test and analysis of algorithme=========================="""


"""
def singleTest(f,f2, N, D, iteration,centre, taille, refParam1,refParam2,C =1., gamma=0.1, n=1, num_cluster=3, fileName = fileName): 
    print("Nouvelle simulation, calculating...")
    removePKLrecursive()
    creatFile(fileName = fileName)
    clearCSV(fileName = fileName)
    depart = Genetique_svm(f, N, centre, forcer = Forcer, C =C, taille =taille, seuil = 0.502, \
                           fileName = fileName, gamma=gamma, num_cluster= num_cluster)
    # seuil = temperature  seuil should be 0. ~ 0.9
    # Conseil d'utilisation seuil = 0.05
    
    
    for j in range(n):
#        depart = Genetique_svm_iter(N, centre, i = j + 1,C =1., taille =taille, gamma=0.8, seuil_filtre = 0.50 ,seuil = 0.0)
        depart = Genetique_svm_iter(f, N, centre, i = j + 1,C =1., taille =taille, gamma=0.8, seuil_filtre = 0.50 ,seuil = 0.0, num_cluster= num_cluster)
    
    deltaGenParam1_0 = D.denormalizeValue(depart[0][0], 0) - refParam1
    deltaGenParam2_0 = D.denormalizeValue(depart[0][1], 1) - refParam2
    deltaGenParam1_1 = D.denormalizeValue(depart[1][0], 0) - refParam1
    deltaGenParam2_1 = D.denormalizeValue(depart[1][1], 1) - refParam2
    deltaGenParam1_2 = D.denormalizeValue(depart[2][0], 0) - refParam1
    deltaGenParam2_2 = D.denormalizeValue(depart[2][1], 1) - refParam2
    
    minCostGen = np.min(f2(depart))                                     
                      
#==============================================================================
#     deltaGenParam1_0 = depart[0][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaGenParam2_0 = depart[0][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#     deltaGenParam1_1 = depart[1][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaGenParam2_1 = depart[1][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#     deltaGenParam1_2 = depart[2][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaGenParam2_2 = depart[2][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#==============================================================================
    
                                         
                                         
                                         
    deltaGen_0 = np.sqrt(deltaGenParam1_0**2 + deltaGenParam2_0**2)
    deltaGen_1 = np.sqrt(deltaGenParam1_1**2 + deltaGenParam2_1**2)
    deltaGen_2 = np.sqrt(deltaGenParam1_2**2 + deltaGenParam2_2**2)
    
    deltaGen = np.min(np.array([deltaGen_0, deltaGen_1, deltaGen_2]))

    
    x0, _, cost0= spsa(f, depart[0], iteration,sa,sc)
    x1, _, cost1= spsa(f, depart[1], iteration,sa,sc)
    x2, _, cost2= spsa(f, depart[2], iteration,sa,sc)
    
    minCostSPSA = np.min(np.array([cost0, cost1, cost2]))
    
    print("================================================================")
    print("Meilleur cost de Genetic : ", minCostGen)
    print("================================================================")
    print("Meilleur cost de SPSA apres Genetic : ", minCostSPSA)
    print("================================================================")
    print("Nombre d'appel Aimsun Genetic : ", N)
    print("================================================================")
    print("Nombre d'appel Aimsun SPSA : ", 2*iteration)
    print("================================================================")
    
    deltaOptParam1_0 = D.denormalizeValue(x0[0], 0) - refParam1
    deltaOptParam2_0 = D.denormalizeValue(x0[1], 1) - refParam2
    deltaOptParam1_1 = D.denormalizeValue(x1[0], 0) - refParam1
    deltaOptParam2_1 = D.denormalizeValue(x1[1], 1) - refParam2
    deltaOptParam1_2 = D.denormalizeValue(x2[0], 0) - refParam1
    deltaOptParam2_2 = D.denormalizeValue(x2[1], 1) - refParam2
    
#==============================================================================
#     deltaOptParam1_0 = depart[0][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaOptParam2_0 = depart[0][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#     deltaOptParam1_1 = depart[1][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaOptParam2_1 = depart[1][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#     deltaOptParam1_2 = depart[2][0] - D.normalizeParamColumnMax_Min(refParam1, 0)
#     deltaOptParam2_2 = depart[2][1] - D.normalizeParamColumnMax_Min(refParam2, 1)
#==============================================================================
                                          
                                          
    deltaOpt_0 = np.sqrt(deltaOptParam1_0**2 + deltaOptParam2_0**2)
    deltaOpt_1 = np.sqrt(deltaOptParam1_1**2 + deltaOptParam2_1**2)
    deltaOpt_2 = np.sqrt(deltaOptParam1_2**2 + deltaOptParam2_2**2)
    deltaOpt = np.min(np.array([deltaOpt_0, deltaOpt_1, deltaOpt_2]))
    
    return deltaGen, deltaOpt, minCostGen, minCostSPSA


def refParam(param):
    if (param=="reactiontime"):
        return 1.2
    if (param=="lookaheaddistance"):
        return 200.
    if (param=="reactiontimeattrafficlight"):
        return 1.
    if (param=="reactiontimevariation"):
        return 1.

def unityTest(M, N, n, iteration,centre, taille, param1, param2, C =1., gamma="auto", fileName = fileName): 
    # N = le nombre d'échantillons, n = le nombre d'itérations, M = nombre de tests voulus
    # param1, param2, lossFunctionType en str. lossFunctionType de la forme "INDEX_L1"
    D = Dataset("data_two_parameters_"+param1+"_"+param2+".csv")
    lossFunction = D.lossFunction(D.INDEX_L2)

    def SimulationAimsun2(X,fileName = fileName): 
        Len = len(X)
        f = open(fileName+".csv", "a")
        writer = csv.writer(f)
        Y = np.zeros(Len)
        for i in range(Len):
#            print(lossFunction(X[i,:].reshape(1,2)))
            Y[i] = lossFunction(X[i,:].reshape(1,2))
            writer.writerow((X[i,0], X[i,1], Y[i]))
        
        f.close()
        return featureNormalize1(Y)
  
    genError = np.ones(M)
    optError = np.ones(M)
    cost_gen = 0
    cost_spsa = 0
    for i in range(M):
        deltaGen, deltaOpt, CostGen, CostSPSA = singleTest(SimulationAimsun2,lossFunction, N, D, iteration,centre, taille, refParam(param1),refParam(param2),C = C, gamma=gamma, n = n, fileName = fileName)
        genError[i]=deltaGen
        optError[i]=deltaOpt
        cost_gen += CostGen
        cost_spsa += CostSPSA
    genError=genError/(2*np.sqrt(taille[0]**2+taille[1]**2))
    optError=optError/(2*np.sqrt(taille[0]**2+taille[1]**2))
    cost_gen /= M
    cost_spsa /= M
    return genError, optError, cost_gen, cost_spsa




removePKLrecursive()
creatFile(fileName = fileName)
clearCSV(fileName = fileName)
Centre = np.array([0.5, 0.5]).reshape(1,2)
taille = np.array([0.5, 0.5])
start =  t()
genError, optError, cost_gen, cost_spsa = unityTest(50, 500, 0, iteration, Centre, taille, "reactiontime","reactiontimeattrafficlight", fileName = fileName, C=100., gamma= 0.09)
end = t()
print("Temps de calcul est = "+str(end-start))

plt.figure(101)
plt.hist(genError, label="Precision de l'algorithme genetique")
plt.legend(loc="best")
plt.figure(102)
plt.hist(optError, label="Precision de l'algorithme SPSA+gen")
plt.legend(loc="best")

print("Pour la distribution de la précision du gen, on a mean="+str(np.mean(genError))+" std="+str(np.std(genError))+" median="+str(np.median(genError)))
print("Pour la distribution de la précision du spsa+gen, on a mean="+str(np.mean(optError))+" std="+str(np.std(optError))+" median="+str(np.median(optError)))
print("Cost moyenne du Genetic "+str(cost_gen) )
print("Cost moyenne du Genetic+SPSA "+str(cost_spsa) )



"""