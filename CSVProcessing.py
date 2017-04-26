# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:38:00 2017

@author: Lucas
"""

import csv
import math
import random
import numpy as np
from scipy.optimize import minimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
"""
A class to process the data exported in a .csv file.
"""

class Dataset:
    
    INDEX_L1 = 0
    INDEX_L2 = 1
    INDEX_GEH = 2
    INDEX_ENTROPY = 3
    INDEX_COEFFREG = 4
    
    NUMBER_OF_PARAMS = 2 #The number of parameters considered.
    NUMBER_OF_LOSSFUNCTIONS = 5 #The number of loss functions considered.
    
    normalisationCoefficients = [0 for i in range(NUMBER_OF_PARAMS + NUMBER_OF_LOSSFUNCTIONS)]  #The coefficients of normalisation, to explicitly compute transformation.
    
    """ Initialization of the class.
    """
    def __init__(self, path, nb_params = 2, nb_lossfunctions = 5):
        
        """ Opening the dataset. """
        print("Opening Dataset ...")
        self.NUMBER_OF_PARAMS = nb_params #The number of parameters considered.
        self.NUMBER_OF_LOSSFUNCTIONS = nb_lossfunctions #The number of loss functions considered.
        f = open(path, "r")
        reader = csv.reader(f)
        self.data = []
        for row in reader:
            if len(row)==0:
                continue
            row = [float(e) for e in row]
            x = row[:self.NUMBER_OF_PARAMS]
            y = row[self.NUMBER_OF_PARAMS: ]
            self.data.append({"x":np.array(x), "y":np.array(y)})
        print("Dataset successfully opened.")
        
        """ Normalize the data. """
        print("Normalising Data ...")
        self.normalizeAll()
        print("Data successfully normalized.")
        
        return
    
    """ Normalize the colomn corresponding to the PARAM_INDEX loss function, using (x - min)/(max - min) formula.
    """
    def normalizeLossColumnMax_Min(self, PARAM_INDEX):
        normalizedData = []
        lossValues = [row["y"][PARAM_INDEX]  for row in self.data]
        maxValue = max(lossValues)
        minValue = min(lossValues)
        self.normalisationCoefficients[PARAM_INDEX + self.NUMBER_OF_PARAMS] = minValue, maxValue
        for i in range(len(lossValues)):
            if (minValue != maxValue):
                normalizedData.append( (lossValues[i] - minValue) / (maxValue - minValue))
            else:
                normalizedData.append(0)
        return normalizedData

    """ Normalize the colomn corresponding to the PARAM_INDEX loss function, using (x - min)/(max - min) formula.
    """
    def normalizeParamColumnMax_Min(self, PARAM_INDEX):
        normalizedData = []
        lossValues = [row["x"][PARAM_INDEX]  for row in self.data]
        maxValue = max(lossValues)
        minValue = min(lossValues)
        self.normalisationCoefficients[PARAM_INDEX] = minValue, maxValue
        for i in range(len(lossValues)):
            if (minValue != maxValue):
                normalizedData.append( (lossValues[i] - minValue) / (maxValue - minValue))
            else:
                normalizedData.append(0)
        return normalizedData
  
    """ Normalize all columns : parameters and lossvalues, using the min-max formula.
    """
    def normalizeAll(self):
        normalizedLossColumns = []
        normalizedParamColumns = []
        for PARAM_INDEX in range(self.NUMBER_OF_LOSSFUNCTIONS):
            normalizedLossColumns.append(self.normalizeLossColumnMax_Min(PARAM_INDEX))
        for PARAM_INDEX in range(self.NUMBER_OF_PARAMS):
            normalizedParamColumns.append(self.normalizeParamColumnMax_Min(PARAM_INDEX))
        for i in range(len(self.data)):
            row = self.data[i]
            row["x"] = np.array([column[i] for column in normalizedParamColumns])
            row["y"] = np.array([column[i] for column in normalizedLossColumns])
        
    """ Denormalize the given value by inversing the max_min formula. 
    """
    def denormalizeValue(self, value, PARAM_INDEX):
        min, max = self.normalisationCoefficients[PARAM_INDEX]
        return min + (max - min) * value
         
    
    """ Modifies self.data to normalize the loss functions, using the specified method.
    """
    def transformData(self, transformationMethod):
        normalizedColumns = []
        for PARAM_INDEX in range(self.NUMBER_OF_LOSSFUNCTIONS):
            normalizedColumns.append(transformationMethod(PARAM_INDEX))
        for i in range(len(self.data)):
            row = self.data[i]
            row["y"] = np.array([column[i] for column in normalizedColumns])
    
    
    """ Transform the column corresponding to the PARAM_INDEX loss function, using a sigmoid.
    """    
    def transformationColumnSigmoid(self,  PARAM_INDEX):
        lossValues = [row["y"][PARAM_INDEX]  for row in self.data]
        listMax = float(max(lossValues))
        transformedData = [1/(1+math.exp(-i/listMax)) for i in lossValues]
        return transformedData
    
        """ Transform the column corresponding to the PARAM_INDEX loss function, using tanh.
    """    
    def transformationColumnTanh(self,  PARAM_INDEX):
        lossValues = [row["y"][PARAM_INDEX]  for row in self.data]
        listMax = float(max(lossValues))
        transformedData = [1/(1+math.exp(-i/listMax)) for i in lossValues]
        return transformedData
    
    """ Transform the column corresponding to the PARAM_INDEX loss function, using standard score normalization.
    """  
    def transformationColumnStandardScore(self,  PARAM_INDEX):
        lossValues = [row["y"][PARAM_INDEX]  for row in self.data]
        listAverage = np.mean(lossValues)
        listStd = np.std(lossValues)
        if listStd != 0:
            transformedData = [(i - listAverage) / listStd for i in lossValues]
        else:
            transformedData = [0 for i in lossValues]
        return transformedData
    
    
    """ Exports the data in path
    """
    def exportData(self, path):
        f = open(path, "w")
        writer = csv.writer(f)

        for row in self.data:
            line = [row["x"][i] for i in range(self.NUMBER_OF_PARAMS)] + [row["y"][i] for i in range(self.NUMBER_OF_LOSSFUNCTIONS)]
            writer.writerow(line)
        f.close()

    
    """ Computes an interpolation function for the PARAM_INDEX loss function, using the scipy library. WORKS FOR 2 PARAMETERS ONLY.
    """
    def bivariateSplineInit(self, PARAM_INDEX, len_grid = 100):
        x1 = np.array([0. for i in range(len_grid)]) #reactionTime
        x2 = np.array([0. for i in range(len_grid)]) #lookaheadDistance
        y = np.ones((len_grid, len_grid)) #loss function table
        counter = 0
        for i in range(len_grid):
            for j in range(len_grid):
                row = self.data[counter]
                x1[i] = row["x"][0]
                x2[j] = row["x"][1]
                y[i][j] = row["y"][PARAM_INDEX]
                counter = counter + 1
        self.interpolationFunction = interpolate.RectBivariateSpline(x1, x2, y)
    
    def bivariateSpline(self, x , PARAM_INDEX):
        
        x1 = x[:,0]
        x2 = x[:,1]
        return self.interpolationFunction.__call__(x1, x2, grid = False)
    
    
    """ Computes an interpolation function for the PARAM_INDEX loss function, using the scipy library. WORKS FOR 3 PARAMETERS ONLY.
    """
    def interpolation3DInit(self, PARAM_INDEX, len_grid = 30):
        x1 = np.array([0. for i in range(len_grid)]) #reactionTime
        x2 = np.array([0. for i in range(len_grid)]) #reactionTimeAtTrafficLight
        x3 = np.array([0. for i in range(len_grid)]) #lookaheadDistance
        y = np.ones((len_grid, len_grid, len_grid)) #loss function table
        counter = 0
        for i in range(len_grid):
            for j in range(len_grid):
                for k in range(len_grid):
                    row = self.data[counter]
                    x1[i] = row["x"][0]
                    x2[j] = row["x"][1]
                    x3[k] = row["x"][2]
                    y[i][j][k] = row["y"][PARAM_INDEX]
                    counter = counter + 1
        self.interpolationFunction =  interpolate.RegularGridInterpolator((x1, x2, x3), y, bounds_error=False, fill_value=1.)
        
    def interpolation3D(self, x , PARAM_INDEX):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        return self.interpolationFunction.__call__(x1, x2, x3)
    
    """ Define a loss function. """
    def lossFunction(self, PARAM_INDEX):
        if self.NUMBER_OF_PARAMS == 2:
            self.bivariateSplineInit(PARAM_INDEX) 
            def function(x):    
                return self.bivariateSpline(x, PARAM_INDEX)
        elif self.NUMBER_OF_PARAMS == 3:
            self.interpolation3DInit(PARAM_INDEX)
            def function(x):
                return self.interpolation3D(x, PARAM_INDEX)
        #General case to complete if needed.
        return function
        
    
    """ Show a 3d graph (2 parameters) for the PARAM_INDEX parameter.
    """
    def plotValues(self, PARAM_INDEX):
        
        nb_values_x1 = 100
        nb_values_x2 = 100
        
        x1 = np.array([0. for i in range(nb_values_x1)]) #reactionTime
        x2 = np.array([0. for i in range(nb_values_x2)]) #lookaheadDistance
        y = np.ones((nb_values_x1, nb_values_x2)) #loss function table
        counter = 0
        for i in range(nb_values_x1):
            for j in range(nb_values_x2):
                row = self.data[counter]
                x1[i] = row["x"][0]
                x2[j] = row["x"][1]
                y[j][i] = row["y"][PARAM_INDEX]
                counter = counter + 1
        
                
        print("Beginning computation")
        X1, X2 = np.meshgrid(x1, x2)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_title('Fonction de perte CoeffReg')
        ax.set_xlabel('Reaction Time')
        ax.set_ylabel('Lookahead Distance')
        surf = ax.plot_surface(X1, X2, y, cmap = cm.RdPu )
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
    
        fig2 = plt.figure()
        proj = plt.contourf(X1, X2, y,
                      alpha=0.8,
                      cmap=plt.cm.bone,
                      extend='both')
        plt.title('Fonction de perte CoeffReg, lignes de niveau')
        plt.xlabel('Reaction Time')
        plt.ylabel('Lookahead Distance')

    
        ligne_niveau = plt.contour(proj, 
                      colors='b',
                      hold='on')
        cbar = plt.colorbar(proj)
        cbar.add_lines(ligne_niveau)
        
        plt.show()