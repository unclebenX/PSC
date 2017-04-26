# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:51:56 2017

@author: Benjamin
"""

import numpy as np
import sqlite3

from scipy import stats

"""
This class implements the different loss function we imagined : 
        -L1 distance
        -L2 distance
        -GEH
        -Entropy
        -Linear regression coefficient
"""

""" The following function retrieves the wanted values from the given database
"""
def fetchData(path, table, attribute):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT " + attribute + " FROM " + table + ";")
    rows = cursor.fetchall()
    #print("Fetched", attribute, np.asarray(rows))
    return np.asarray(rows)


""" Implementation of L1 distance, L2 distance and GEH distance.
"""
def L1distance(x,y):
    return abs(x-y)

def L2distance(x,y):
    return (x-y)**2

def GEHdistance(x,y):
    if(x+y<0.1):
        return 0.
    return (np.abs(x-y))/np.sqrt((x+y)/2)

def attrDistance(path1, path2, table, attribute, distance):
    data1 = fetchData(path1, table, attribute)
    data2 = fetchData(path2, table, attribute)
    dist = 0.
    for i in range(len(data1)):
        dist += distance(data1[i][0], data2[i][0])
    return dist

def Loss(path1, path2, distance):
    distMESECT = 0.
    Attr_MESECT = ["flow", "ttime", "density"]
    #Attr_MESECT = ["flow", "count", "input_flow", "input_count", "ttime", "dtime", "wtimeVQ", "dtimeTtime", "speed", "spdh", "flow_capacity", "density", "qvnbvehs", "qmean","qmax", "qvmean",  "qvmax", "travel", "traveltime","lane_changes", "total_lane_changes"]
    for attribute in Attr_MESECT:
        distMESECT += attrDistance(path1, path2, "MESECT", attribute, distance)
        #print("(LOSS) Distance ", attribute, " : ", distMESECT)
    return distMESECT


""" Implementation of Entropy.
"""

def EntropyNorm(p):
    if p<0.001:
        return 0
    return -p * np.log(p) 

def attrNormalization(path1, path2, table, attribute):
    data1 = fetchData(path1, table, attribute)
    data2 = fetchData(path2, table, attribute)
    a = np.min(np.abs(data1 - data2))
    b = np.max(np.abs(data1 - data2))
    normalizedDistance = []
    for i in range(len(data1)):
        if (b != a):
            normalizedDistance.append((np.abs(data1[i][0] - data2[i][0]) - a) / (b - a))
        else:
            normalizedDistance.append(0)
    return np.array(normalizedDistance)
            
def normalizedAttrNorm(normalizedAttribute, norm):
    dist = 0.
    for i in range(len(normalizedAttribute)):
        dist += norm(normalizedAttribute[i])
    return dist

def EntropyLoss(path1, path2):
    distMESECT = 0.
    Attr_MESECT = ["flow", "ttime", "density"]
    #Attr_MESECT = ["flow", "count", "input_flow", "input_count", "ttime", "dtime", "wtimeVQ", "dtimeTtime", "speed", "spdh", "flow_capacity", "density", "qvnbvehs", "qmean","qmax", "qvmean",  "qvmax", "travel", "traveltime","lane_changes", "total_lane_changes"]
    for attribute in Attr_MESECT:
        normalizedAttribute = attrNormalization(path1, path2, "MESECT", attribute)
        distMESECT += normalizedAttrNorm(normalizedAttribute, EntropyNorm)
    return (distMESECT)


""" Implementation of Linear Regression Coefficient.
"""

def CoeffRegLoss(path1, path2):
    coeff=0.0
    Attr_MESECT = ["flow", "ttime", "density"]
    #Attr_MESECT = ["flow", "count", "input_flow", "input_count", "ttime", "dtime", "wtimeVQ", "dtimeTtime", "speed", "spdh", "flow_capacity", "density", "qvnbvehs", "qmean","qmax", "qvmean",  "qvmax", "travel", "traveltime","lane_changes", "total_lane_changes"]
    for attribute in Attr_MESECT:
        data1 = fetchData(path1, "MESECT", attribute)
        data2 = fetchData(path2, "MESECT", attribute)
        a,b,r,p,std_err = stats.linregress([d[0] for d in data1], [d[0] for d in data2])
        coeff = coeff + r
    loss = np.abs(coeff/len(Attr_MESECT) - 1)
    return loss