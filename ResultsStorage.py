# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:43:35 2017

@author: Benjamin
"""

import RemoteAimsun as remote
import AimsunLossFunctions, csv
import numpy as np


def main():
    aim = remote.Aimsun()
    f = open("data_4_parameters.csv", "a")
    writer = csv.writer(f)
    
    #print(LossFunction([0.95,0.,0.,500.],aim),LossFunction([0.95,0.,0.,501.],aim),LossFunction([0.95,0.,0.,499.],aim)
    reactionTime = [1.2] #np.arange(0.9, 1.4, 0.05)  # 1.2
    reactionTimeAtTraficLight = [1.0] # np.arange(0.5, 1.5, 0.05)  # 1.0
    lookAheadDistance = [200.0] #np.arange(150.0,210.0, 0.5)  # 200.
    reactionTimeVariation = [1.0]#np.arange(.7, 1.3, 0.07) #
    
    
    for i in range(len(reactionTime)):
        for j in range(len(reactionTimeVariation)):
            for k in range(len(reactionTimeAtTraficLight)):
                    
                        #params_ref = [1.05,0.1,1.5,500.]
                        params = [reactionTime[i], reactionTimeAtTraficLight[j], lookAheadDistance[k], 200.] # + [1. for i in range(101)]
                        aim.setParams(params)
                        aim.simulate()
                        loss_GEH = AimsunLossFunctions.LossFunction_GEH(params, aim)
                        loss_L1 = AimsunLossFunctions.LossFunction_L1(params, aim)
                        loss_L2 = AimsunLossFunctions.LossFunction_L2(params, aim)
                        print('Valeurs de perte :', loss_GEH, loss_L1, loss_L2)
                        writer.writerow((reactionTime[i], reactionTimeAtTraficLight[j], lookAheadDistance[k], 200., loss_GEH, loss_L1, loss_L2))
    
    f.close()
    aim.interrupt()
main()