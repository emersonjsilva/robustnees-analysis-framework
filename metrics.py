#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:05:56 2020

@author: emerson
"""

#No data -9999

import numpy as np
import math as mt



def validationOfData(observed, predicted):
    if len(observed) == len(predicted):
        observed = np.array(observed)
        predicted = np.array(predicted)
        return 1, observed, predicted
    return 0, 0, 0    

def me(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return sum(observed - predicted) / len(observed)
    return -9999

def mae(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return sum(abs(observed - predicted)) / len(observed)
    return -9999

def mse(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return sum((observed - predicted) ** 2) / len(observed)
    return -9999

def rmse(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return mt.sqrt(sum((observed - predicted) ** 2) / len(observed))
    return -9999

def nash(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return 1 - sum((observed - predicted)**2) / sum((observed - np.mean(observed)) ** 2)
    return -9999
    
def accuracyOfMean(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return 1 - abs((np.mean(observed) - np.mean(predicted)) / np.mean(observed))
    return -9999

def accuracyOfMedian(observed, predicted):
    test, observed, predicted = validationOfData(observed, predicted)
    if test == 1:
        return 1 - abs((np.median(observed) - np.median(predicted)) / np.median(observed))
    return -9999

def precision(predicted):
    return 1 - abs((np.std(predicted) / np.mean(predicted)))

def robustness(entrada, saida):
    test, entrada, saida = validationOfData(entrada, saida)
    if test == 1:
        return (np.std(entrada) - np.std(saida)) / np.std(entrada)
    return -9999

def accuracyOfMeanSingle(observed, predicted):
    return 1 - abs((np.mean(observed) - np.mean(predicted)) / np.mean(observed))


def accuracyOfMedianSingle(observed, predicted):
    return 1 - abs((np.median(observed) - np.median(predicted)) / np.median(observed))

def noiseLevelPercetageMean(serie, levelPorcentage):
    return np.mean(serie) * (levelPorcentage / 100)

def noiseLevelPercetageMax(serie, levelPorcentage):
    return np.max(serie) * (levelPorcentage / 100)
