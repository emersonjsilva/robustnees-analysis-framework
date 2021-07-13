#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:52:49 2019

@author: emerson
"""

import numpy as np
import pandas as pd

class gauss:
    
    def __init__(self):
        self.seed = 1
        np.random.seed(self.seed)
    
    def setSeed(self, seed_):
        self.seed = seed_
        np.random.seed(self.seed)
        
        
    def gaussiana(self, media, std_, len_):
       return np.random.normal(loc=media, scale=std_, size=len_) 
      
    #Fim da classe
    
    
    
'''
    Função para criar vetor de distribuição gaussiana e combinar no modelo y = ax+b


def y_gaussiana(media, std, len_, seed, ds_nome):
   np.random.seed(seed)
   a = np.random.normal(loc=media, scale=std, size=len_) 
   b = np.random.normal(loc=media, scale=std, size=len_) 
   x = np.random.normal(loc=media, scale=std, size=len_)
   df = pd.DataFrame(np.transpose([a, x, b]))      
   df.to_csv(ds_nome+'_inputs.csv', sep=',', header=['a', 'x', 'b'], index= False) 
   
   y = a * x + b
               
               
   df_ = pd.DataFrame(np.transpose([a, x, b, y]))       
   df_.to_csv(ds_nome+'.csv', sep=',', header=['a', 'x', 'b', 'y'], index= False)
   
#Fim da função
'''  