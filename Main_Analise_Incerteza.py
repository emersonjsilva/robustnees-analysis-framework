#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:00:35 2020

@author: emerson
"""

'''
    Pendencias:
        1. Salvar todos os dataframes em arquivos
        2. Ver robustez, análise dos valores estacao/saída da rede
        3. Ver limite minimo do grafico da incerteza (ok)
        4. Implementar função para ver quantas vezes o valor medio acontece (nenhuma)
        
'''

# Bibliotecas padrão
import numpy as np
import random as rd
import pandas as pd
from scipy import stats 
import matplotlib
import matplotlib.pyplot as plt 
import math as mt

# Bibliotecas desenvolvidas 
from metrics import *
from plots import *
from gaussiana import *
from Modelo_Hidrologico import *
from datetime import datetime



  
##############################################################################
#                           Parâmetros do modelo
# Treinar modelo = 1, não treinar = 0
trainModel = 0

# Carregar pesos
loadWeights = 1

typeOfCase = 'extremo'
seriesData = 'BaciaConselheiroPaulino'
pathSeries = 'Serie'

# Semente
seed = 1

# Plotar graficos
plot__ = 0

#                               OBSERVAÇÃO
# Para alternar entre os modelos de 15 e 120 minutos, altear linha 40 do arquivo
# "Modelo_Hidrologico.py" a variável LAG_MIN (15 minutos lag_min = 1, 120 minutes
# lag_min = 8)
##############################################################################


print(DIR)

###############################################################################
#
# Função para o cálculo dos momentosEstatisticosSerie estatísticos de cada estacao
#
# Vetor retorno: 0 = min, 1 = max, 2 = mediana, 3 = media 
#
###############################################################################
MIN = 0
MAX = 1
MEDIAN = 2
MEAN = 3

def statsMomentsOfSeries(series):
    moments = []    
    arraySeries = np.array(series)
    for colStats in range(len(arraySeries[0,:])):
        moments.append([np.min(arraySeries[:,colStats]),
                    np.max(arraySeries[:,colStats]),
                    (np.max(arraySeries[:,colStats]) - np.min(arraySeries[:,colStats])) / 2, np.std(arraySeries[:,colStats]),
                    np.mean(arraySeries[:,colStats])])        
    return np.array(moments)


###############################################################################
#
# Função para o cálculo dos momentos estatísticos de cada estacao
#
# Vetor retorno: valor estações tempo t, observado t + 1
#
###############################################################################

def nearCaseFinder(Series, observed, value):
    t = 0
    t_ = -1
    error = np.abs(observed[t] - value)
    for t in range(len(Series[ : , 0])):
        if(error > np.abs(observed[t] - value)):            
            t_ = t
            error = np.abs(observed[t] - value )             
    
    return t_, Series[t_, : ], observed[t_, FORECAST_STATION]

def nearCaseFinderGraterZero(Series, observed, value):
    t = 0
    t_ = -1
    error = np.abs(observed[t] - value)
    for t in range(len(Series[ : , 0])):
        if(error > np.abs(observed[t] - value)):            
            test = True
            for station in range(len(Series[0, : ])-1):
                if(not (Series[t, station] > 0)):
                    test = False
                    
            if(test):
                t_ = t
                error = np.abs(observed[t] - value )             
    
    return t_, Series[t_, : ], observed[t_, FORECAST_STATION]


def maxSeriesInTimeT(series, observed, station):    
    arraySeries = np.array(series)  
    observed = np.array(observed)  
    
    if(typeOfCase == 'extremo'): 
        #print('extremo')
        station = INTEREST_STATION # fixar todas no extremo exultorio
        
    # max value in station
    if(typeOfCase == 'extremo na estação' or typeOfCase == 'extremo'):
        #print('extremo na estação')
        maximum = arraySeries[0, station]
        t_ = 0
        for t in range(len(arraySeries[ : , station])):        
            if(arraySeries[t, station] > maximum):
                t_ = t
                maximum = arraySeries[t, station]   
                
    # Average case value in observed level station       
    elif(typeOfCase == 'medio'):
        #print('medio')
        averageCase = (max(arrayObservedData[:, 0]) + min(arrayObservedData[:, 0])) /2   
        #averageCase = np.mean(arrayObservedData)
        t_, ret1, ret2 = nearCaseFinder(arraySeries, observed, averageCase)
    
    # Minimun value in in observed level station
    elif(typeOfCase == 'minimo'):
        #print('minimo')
        t_, ret1, ret2 = nearCaseFinder(arraySeries, observed, 0)
            
    inputsReturn = arraySeries[t_, : ]
    observedRet = observed[t_, FORECAST_STATION]
    
    return  inputsReturn, observedRet, t_

###############################################################################
#
# Função para o verificar se uma faixa de valores ocorreram
#
#
###############################################################################

def valuesInSeries(series, values, tolerance): 
    arraySeries = np.array(series)  
    arrayValue = np.array(values)
    arrayTolerance = np.array(tolerance)
    listIndex = []
   
    counter = 0
    for t in range(len(arraySeries[ : , 0])): 
        test = True
        for station in range(len(arraySeries[0, : ])):        
            if(arraySeries[t, station] > (arrayValue[station] + arrayTolerance[station]) or
               arraySeries[t, station] < (arrayValue[station] - arrayTolerance[station])):
                test = False
        if(test == True):
            counter = counter + 1
            listIndex.append(t)
    return  counter, listIndex

###############################################################################
#
# Função para o verificar quantidade de valores menores que zero
#
#
###############################################################################

def negativeValuesPercent(series): 
    arraySeries = np.array(series)  
   
    counter = 0
    for t in range(len(arraySeries)): 
        if(arraySeries[t] < 0):
            counter = counter + 1
    percent = counter / len(arraySeries)
    return  counter, percent



###############################################################################
#
# Função para o cálculo dos momentosEstatisticosSerie estatísticos da saída
#
# Vetor retorno: 0 = media, 1 = std 
#
###############################################################################

def mean_std(series):
    listReturn = []
    series = np.array(series)
    numCols = len(series[0])
    for i in range(numCols):
        listReturn.extend([np.mean(series[i,:])])
        listReturn.extend([np.std(series[i,:])])
    return listReturn


###############################################################################
#
# Modelo hidrológico neural
#    
###############################################################################


dataFrameData = readCsvDataset(pathSeries, seriesData)

# Considera como saida todas as entradas para o tempo t + 1

inputLabels = ['chuva_Olaria', 'nivel_Olaria',
          'chuva_VendaDasPedras', 'nivel_VendaDasPedras',
          'chuva_Suspiro', 'nivel_Suspiro',
          'chuva_Ypu', 'nivel_Ypu',
          'chuva_ConselheiroPaulino', 'nivel_ConselheiroPaulino']

observedLabels = ['nivel_ConselheiroPaulino']

inputNames = ['Chuva Olaria', 'Nível Olaria',
          'Chuva Venda Das Pedras', 'Nível Venda Das Pedras',
          'Chuva Suspiro', 'Nível Suspiro',
          'Chuva Ypu', 'Nível Ypu',
          'Chuva Conselheiro Paulino', 'Nível Conselheiro Paulino']

observedNames = ['Nível Conselheiro Paulino']

listStationsMetrics = ['Precipitação [mm]', 'Nível [m]',
          'Precipitação [mm]', 'Nível [m]',
          'Precipitação [mm]', 'Nível [m]',
          'Precipitação [mm]', 'Nível [m]',
          'Precipitação [mm]', 'Nível [m]']

listObservedMetrics = ['Nível [m]']

# Organiza as colunas por nome
dataFrameInputData = dataFrameData[inputLabels]
arrayInputDataRaw = np.array(dataFrameData[inputLabels])

# Considera como saida os dados de nível
dataFrameObservedData = dataFrameData[observedLabels]

plotTimeSeries(dataFrameObservedData, "Nível Concelheiro Paulino",
                  'Data', 'Nível [m]', DIR + "0.0 - Dataset.png")

# Remove o ultimo registro LAG_MIN
dataFrameInputData = dataFrameInputData[ : -LAG_MIN]

# Remove primeiro registro LAG_MIN (previsão t + 1)
dataFrameObservedData = dataFrameObservedData[LAG_MIN : ]

arrayInputData = np.array(dataFrameInputData)
arrayObservedData = np.array(dataFrameObservedData)


# Treinamento ou carregamento dos pesos
print("0 - Init model")
arrayNormInputs, arrayNormObserved, scalerInput, scalerObserved, model = hydrologicalModel (trainModel,
                                        loadWeights,
                                        dataFrameInputData, 
                                        dataFrameObservedData, plot__)

dataStationsSetTest = np.array(dataFrameInputData)

# Set of Tests
arrayDenormInputs = scalerInput.inverse_transform(arrayNormInputs)

#dataStationsSetTest = dataStationsSetTest[:4109, :]
dataStationsSetTest =  scalerInput.transform(dataStationsSetTest)

lenTrainSetInput = np.int32(np.around(len(arrayNormInputs)*0.5))

plot__ = 1
returnedObserved, predictionsReturned = predicao(scalerInput, scalerObserved,
                                 model, arrayNormInputs, arrayNormObserved, plot__,
                                 lenTrainSetInput, "Previsão " + str(LAG_MIN * 15) + " minutos todo o dado")


x_ = arrayNormInputs
y_ = arrayNormObserved

# Divide o numero de amostras na metade
len_x_train = np.int32(np.around(len(x_)*0.5))
len_y_train = np.int32(np.around(len(y_)*0.5))    
# Separa treino 50% e testes 50% (para calibrar modelo)
x_train_, x_test, y_train_, y_test = x_[0:len_x_train], x_[len_x_train+1:], y_[0:len_x_train], y_[len_x_train+1:]
 
 


returnedObservedTest, predictionsReturnedTest = predicao(scalerInput, scalerObserved,
                                 model, x_test, y_test, plot__,
                                 lenTrainSetInput, "Previsão " + str(LAG_MIN * 15) + " minutos dados de teste")



predict_, norm_prediction = predicao_(scalerInput, scalerObserved, model, dataStationsSetTest)




###############################################################################
#
# 0.1 - Climatological consistency check function   
#
###############################################################################


print("0.1 - Climatological consistency check function")
listMeanInputs = []
tbLabels = []

# Catch the moments of inputs
moments = statsMomentsOfSeries(dataFrameInputData)
listMeanInputs.append(np.transpose(moments[ : , MEAN]))

# Save means of inputs
dataFrameTableCSV = pd.DataFrame(data = np.transpose([inputNames, moments[ : , MEAN]]))
colsName = ["Estação", "Média"]
dataFrameTableCSV.columns = colsName    
dataFrameTableCSV.to_csv (DIR + '0.1 Consistencia médias entrada.csv',
                          index = False, header = True)

# Feed model with means of inputs
listMeanInputs = scalerInput.transform(listMeanInputs)
# Teste no modelo
prediction, norm_prediction = predicao_(scalerInput, scalerObserved, model, listMeanInputs)

# Save table with consistence check
meanObserved =  np.float64(moments[INTEREST_STATION, MEAN])
meanPrediction = np.float64(np.mean(prediction))
dataFrameTableCSV1 = pd.DataFrame(data = np.transpose([[meanObserved], [meanPrediction]]))
colsName1 = ["Média observado", "Média previsão"]
dataFrameTableCSV1.columns = colsName1    
dataFrameTableCSV1.to_csv (DIR + '0.1 Consistencia médias previsão.csv',
                          index = False, header = True)

# How many times the mean values appear in the series
tolerance = [0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01]
print(valuesInSeries(dataFrameInputData, moments[ : , MEAN], tolerance))


###############################################################################
#
# 0.2 - Plota o histograma das estações
#
###############################################################################

plotStations(dataFrameInputData, inputNames, listStationsMetrics, 
             DIR + "0.2 histograma das estações ")

###############################################################################
#
# 1 - Static Analysis
#
###############################################################################
print("1 - Statis Analysis fixed std")
# Verifica os momentos estatisticos das entradas
moments = statsMomentsOfSeries(dataFrameInputData)

# Quantidade de amostras do ruido
lenSamples = 1000

# Distribuição
genRandonNum = gauss() 

#tamanho das amostras
listStdDeviation = np.linspace(0.1, 1.0, num = 20)
#desvio

# Abre arquivo e sobreescreve
CurrentDateTime = datetime.now()
CurrentDateTimeText = CurrentDateTime.strftime('%d/%m/%Y %H:%M')    
arq = open(DIR + '1.0 - robustez.csv', 'w')
arq.write("\nData hora," + CurrentDateTimeText + "\n")
mon = moments[ : , MEAN]

# Inicia processamento
for stdDeviation in listStdDeviation: # Iterate stds
    matrixMetrics = []
    matrixIndexes = []
    
    for station in range(len(moments)): # Each column of moments refer to one station
        mon, obsv, tt_ = maxSeriesInTimeT(dataFrameInputData, dataFrameObservedData, station) #usar para valores extremos
        arrayMom = np.array(mon) 
        meanStation = arrayMom[station]
        genRandonNum.setSeed(seed)
        arrayOfNoise = genRandonNum.gaussiana(meanStation, stdDeviation, lenSamples)
        arrayInputDataWithNoise = []
    
        # Mount input dataset with noise once station
        for noiseIndex in range(lenSamples):
                arrayMom[station] = arrayOfNoise[noiseIndex]
                arrayInputDataWithNoise.append(arrayMom.copy())
                
        arrayInputDataWithNoise = np.array(arrayInputDataWithNoise)
        arrayInputDataWithNoiseNorm = scalerInput.transform(arrayInputDataWithNoise)
        
        predict_, norm_prediction = predicao_(scalerInput, scalerObserved, model, arrayInputDataWithNoiseNorm)
#        if(station == len(moments) - 1):        
#            valores = pd.DataFrame(np.transpose([arrayOfNoise, predict_[: , FORECAST_STATION]]))
#            valores.columns = ['chuva_Olaria', ' nivel ConselheiroPaulino']
#            valores.to_csv("valores_nivel_" + str(stdDeviation) + ".csv")
            
        matrixMetrics = []
        matrixMetrics = [np.mean(predict_[: , FORECAST_STATION]),
                    np.std(predict_[: , FORECAST_STATION]),
                    robustness(arrayInputDataWithNoise[: , station],
                             predict_[: , FORECAST_STATION])]
        matrixIndexes.append(matrixMetrics)        
       
    arrayMatrixIndexes = np.array(matrixIndexes)
    dataFrameTableCSV = pd.DataFrame(data = arrayMatrixIndexes, index = inputNames)
    col_ = ['Média previsão','Desvio padrão previsão', 'Robustez']
    #col_.extend(observedLabels)
    dataFrameTableCSV.columns = col_    
    dataFrameTableCSV.to_csv (DIR + '1.0 Analise estatica ' + str(stdDeviation)
                              + " " + typeOfCase + ".csv",
                              index = True, header = True)

    arq.write("1.0 Desvio," + str(stdDeviation) + ", tamanho " + str(lenSamples) + "\n")
    arq.write(dataFrameTableCSV.to_csv())
    arq.write("\n")
arq.close()    

#dataFrameTableCSV.to_csv('ruido_std_' + str(stdDeviation) + '_estatico_uma_saida.csv')

# Fim 1 - Análise estática

#parar

#typeOfCase = 'extremo na estação'
#typeOfCase = 'extremo'
#typeOfCase = 'medio'
#typeOfCase = 'minimo'

mon, obsv, tt_ = maxSeriesInTimeT(dataFrameInputData, dataFrameObservedData, 0) #usar para valores extremos
        
print(mon)
print(obsv)
print(tt_)



###############################################################################
#
# 2 - Análise estática variando std
#
# Valores médios das estações
#
# Definir o desvio padrão desvio e executar
#
###############################################################################
print("2 - Statis Analysis varying  std")
# Para armazenar os dados
lineStd = []
matrixStd = []
lineRobustness = []
matrixRobustness = []
lineMeanAccuracy = []
tableMeanAccuracy = []
#max_min_norm = []

plotHist = 0

# Tamanho das amostras
lenSamples = 1000
genRandonNum = gauss()
# Definir desvio inicial, final e quantidade de amostras
listStdDeviation = np.linspace(0.1, 2.0, num = 50)
#listStdDeviation = [0.01, 0.1, 1.0]
# Verifica os momentosEstatisticosSerie das entradas
moments = statsMomentsOfSeries(dataFrameInputData)
mon = moments[ : , MEAN]

for stdDeviation in listStdDeviation:#range(1, 11):# para cada desvio
    lineStd = [np.float64(stdDeviation)]
    lineRobustness = [np.float64(stdDeviation)]    
    lineMeanAccuracy = [np.float64(stdDeviation)]
    lin_max_min_norm = []

    for station in range(len(inputNames)):#para cada estacao   
        mon, obsv, tt_ = maxSeriesInTimeT(dataFrameInputData, dataFrameObservedData, station) #usar para valores extremos
        #usar para valores extremos (comentar linha para média)
        arrayMom = np.array(mon)
        meanStation = arrayMom[station]
        genRandonNum.setSeed(seed)
        arrayOfNoise = genRandonNum.gaussiana(meanStation, stdDeviation, lenSamples)
        arrayInputDataWithNoise = []
    
        for noiseIndex in range(lenSamples):#para cada arrayOfNoise gerado
                arrayMom[station] = arrayOfNoise[noiseIndex]
                arrayInputDataWithNoise.append(arrayMom.copy())#monta linha a linha
                
        arrayInputDataWithNoise = np.array(arrayInputDataWithNoise)
        arrayInputDataWithNoiseNorm = scalerInput.transform(arrayInputDataWithNoise)
        
        # Previsão
        predict_, norm_prediction = predicao_(scalerInput, scalerObserved, model, arrayInputDataWithNoiseNorm)
        lineRobustness.extend([robustness(arrayInputDataWithNoise[: , station],
                                      predict_[: , FORECAST_STATION])])

        lineMeanAccuracy.extend([accuracyOfMean([obsv],
                            [np.mean(predict_[: , FORECAST_STATION])])]) #conselheiro paulino
      
        #matrixStd.extend([np.std(predict_[: , FORECAST_STATION])])
        lineStd.extend([np.std(predict_[: , FORECAST_STATION])]) #conselheiro paulino
        lin_max_min_norm.extend([np.min(norm_prediction),
                                 np.max(norm_prediction)])
        
        
        if(plotHist == 1):
            # 2.0.3 histograma das estações #############################################  
            plt.close('all')   
            plt.figure(figsize = (10.5, 5.25))
            plt.rcParams.update({'font.size': 15})
            #plt.title('Distribuição em função da desvio de estacao')
            plt.subplot(121)
            plt.title('Estação \"' + np.str(inputNames[station])
                      + '\" desvio = ' + np.str(stdDeviation))
            
            n1, bins1, patches1 = plt.hist(arrayOfNoise, bins = 'auto',
                                 histtype = 'stepfilled',
                                 facecolor = 'lime')    
            inputMean = np.average(arrayOfNoise)
            inputMedian = np.median(arrayOfNoise)
            #print(np.where(n1 == np.amax(n1)))
            jj = np.where(n1 == np.amax(n1))
            #print(j)
            ii = np.int(np.double(jj[0][0]))    
            inputMode = (bins1[ii] + bins1[ii + 1]) / 2
            plt.axvline(inputMean, color = 'blue', linewidth = 2,label = 'Média [' +"{:.2f}".format(inputMean) +']', alpha = 0.5)
            plt.axvline(inputMode, color = 'red', linewidth = 2, label = 'Moda [' +"{:.2f}".format(inputMode) +']', alpha = 0.5)
            plt.axvline(inputMedian, color = 'green', linewidth = 2, label = 'Mediana [' +"{:.2f}".format(inputMedian) +']', alpha = 0.5)
            plt.xlabel(listStationsMetrics[station])
            plt.ylabel('Frequência')        
            plt.xticks(rotation=45)
            plt.legend(loc = 'best')   
            
            
            plt.subplot(122)
            plt.title('Previsão Nível \"Conselheiro\nPaulino\" desvio = '
                      + np.str(np.std(predict_[: , FORECAST_STATION])))
            
            # Caso o desvio padrão seja igual a zero, ajusta o plot
            if(np.std(predict_[: , FORECAST_STATION]) < 0.001):
                xmin_ = np.mean(predict_[: , FORECAST_STATION]) - 3
                xmax_ = np.mean(predict_[: , FORECAST_STATION]) + 3
                bins1 = np.linspace(xmin_, xmax_, num = len(bins1))
               
            n2, bins2, patches2 = plt.hist(predict_[: , FORECAST_STATION],
                                           bins = 'auto', histtype = 'stepfilled',
                                           facecolor = 'lime')
            meanPrediction = np.average(predict_[: , FORECAST_STATION])
            medianPrediction = np.median(predict_[: , FORECAST_STATION])
            jj = np.where(n2 == np.amax(n2))      
            ii = np.int(np.double(jj[0][0]))    
           
            modePrediction = (bins2[ii] + bins2[ii + 1]) / 2
            plt.axvline(meanPrediction, color = 'blue', linewidth = 2,label = 'Média [' +"{:.2f}".format(meanPrediction) +']', alpha = 0.5)
            plt.axvline(modePrediction, color = 'red', linewidth = 2, label = 'Moda [' +"{:.2}".format(modePrediction) +']', alpha = 0.5)
            plt.axvline(medianPrediction, color = 'green', linewidth = 2, label = 'Mediana [' +"{:.2f}".format(medianPrediction) +']', alpha = 0.5)
            plt.axvline(obsv, color = 'black', linewidth = 1, label = 'Observado [' +"{:.2f}".format(obsv) +']', linestyle = '--')
            plt.xlabel('Nível [m]')
            plt.ylabel('Frequência')        
            plt.xticks(rotation=45)
            plt.legend(loc = 'best')
            plt.savefig(np.str(DIR + "2.0.3 histograma das estações "
                               + np.str(inputNames[station]) + " desvio " 
                               + np.str(stdDeviation) + " " + typeOfCase
                               + ".png"), bbox_inches = 'tight',
                        dpi = imageResolution)
            plt.show()
        

    matrixStd.append(np.array(lineStd))
    matrixRobustness.append(np.array(lineRobustness))   

    tableMeanAccuracy.append(np.array(lineMeanAccuracy))
    #max_min_norm.append(np.array(lin_max_min_norm))

#max_min_norm = np.array(max_min_norm)
matrixRobustness = np.array(matrixRobustness)   

tableMeanAccuracy = np.array(tableMeanAccuracy)   
matrixStd = np.array(matrixStd)
tableStds = pd.DataFrame(data = matrixStd)
y_cols = ['std']
matrixStd = np.array(matrixStd)
tableStds = pd.DataFrame(data = matrixStd)
col_s = ['Desvio padrão perturbação']
col_s.extend(inputNames)
tableStds.columns = col_s
tableStds.to_csv (DIR + '2.1 Analise estatica variado std '+ typeOfCase +'.csv',
                    index = False, header = True)

col_s = ['Acuracia']
col_s.extend(inputNames)
csvAccuracy = pd.DataFrame(data = tableMeanAccuracy)
csvAccuracy.columns = col_s
csvAccuracy.to_csv (DIR + '2.2 Acurácia perturbacao '+ typeOfCase +'.csv',
                    index = False, header = True)

col_s = ['Robustez']
col_s.extend(inputNames)
csvRobustness = pd.DataFrame(data = matrixRobustness)
csvRobustness.columns = col_s
csvRobustness.to_csv (DIR + '2.3 Robustez perturbacao '+ typeOfCase +'.csv',
                    index = False, header = True)

# Salvar em csv

col_s = ['Desvio padrão perturbação']
col_s.extend(inputNames)

tableStds = pd.DataFrame(data = tableStds)
tableStds.columns = col_s
tableStds.to_csv (DIR + '2.3 saida_desvio_perturbacao_'+ typeOfCase +'.csv',
                    index = False, header = True)


# 2.0 Plot saida_desvio_perturbacao_medias ####################################
plt.close('all')
y_cols = ['std']
y_cols.extend(inputNames)
fig, ax = plt.subplots()
ax.set(xlabel = '$\sigma$ ruído de entrada', ylabel = '$\sigma$ previsão',
       title = 'Previsão de nível para\nperturbação no ' + typeOfCase + ' das estações')
for station in range(1, len(tableStds.iloc[0, :])):
    ax.plot(tableStds.iloc[: , 0], tableStds.iloc[: , station], label = y_cols[station])
ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
          ncol = 1, fancybox = True, shadow = True,
          title = 'Previsão de nível para\nperturbação para ' + typeOfCase + '\ndas estações')

fig.savefig(DIR + "2.0 saida_desvio_perturbacao log-log " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()


# 2.1 robustez_perturbacao_medias #############################################
plt.close('all')
fig, ax = plt.subplots()
ax.set(xlabel = '$\sigma$ ruído de entrada', ylabel = 'Robustez', 
       title = 'Robustez nas estações\nperturbação para ' + typeOfCase 
                                   +' das estações')
for station in range(1, len(tableStds.iloc[0, :])):
    ax.plot(matrixRobustness[: , 0], matrixRobustness[: , station], label = y_cols[station])
ax.grid()
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
          ncol = 1, fancybox = True, shadow = True,
          title = 'Previsão de nível para\nperturbação em cada estação')

fig.savefig(DIR + "2.1 robustez_perturbacao_ lin-lin" + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()


# 2.2 acuracia_perturbacao ####################################################
  
plt.close('all')
fig, ax = plt.subplots()
ax.set(xlabel = '$\sigma$ ruído de entrada', ylabel = 'Acurácia', 
       title = 'Acurácia nas estações\nperturbação para a ' + typeOfCase
                   + ' das estações')
for station in range(1, len(tableStds.iloc[1, :])):
    ax.plot(tableMeanAccuracy[: , 0], tableMeanAccuracy[: , station], label = y_cols[station])
ax.grid()
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
          ncol = 1, fancybox = True, shadow = True,
          title = 'Previsão de nível para\nperturbação em cada estação')

fig.savefig(DIR + "2.2 acuracia_perturbacao_lin-lin " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()


# 2.0.1 Plot saida_desvio_perturbacao_medias ##################################
for station in range(1, len(tableStds.iloc[0, :])):
    plt.close('all')
    y_cols = ['std']
    y_cols.extend(inputNames)
    fig, ax = plt.subplots()
    ax.set(xlabel = '$\sigma_{ru\´ido}', ylabel = '$\sigma_{previs\~o}',
           title = 'Previsão de nível para\nperturbação para ' + typeOfCase 
           + ' da estação ' + y_cols[station])
    
    ax.plot(tableStds.iloc[: , 0], tableStds.iloc[: , station], label = y_cols[station])
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
              ncol = 1, fancybox = True, shadow = True,
              title = 'Estação')
    
    fig.savefig(DIR + "2.0.1 saida desvio perturbacao " + typeOfCase + " para estação " 
                + y_cols[station] + " log-log.png",
                bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    # 2.0.1 Plot saida_desvio_perturbacao #########################################
    plt.close('all')
    y_cols = ['std']
    y_cols.extend(inputNames)
    fig, ax = plt.subplots()
    ax.set(xlabel = 'Desvio Padrão estaçao', ylabel = 'Desvio Padrão previsão',
           title = 'Previsão de nível para\nperturbação para ' + typeOfCase 
           + ' da estação ' + y_cols[station])
    
    ax.plot(tableStds.iloc[: , 0], tableStds.iloc[: , station], label = y_cols[station])
    ax.grid()
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
              ncol = 1, fancybox = True, shadow = True,
              title = 'Estação')
    
    fig.savefig(DIR + "2.0.1 saida_desvio_perturbacao_" + typeOfCase + "_para_estação_" 
                + y_cols[station] + "_xlog.png",
                bbox_inches = 'tight', dpi = imageResolution)
    plt.show()

    # 2.0.2 robustez no em função do std ##########################
       
    plt.close('all')
    #plt.figure(figsize = (12.5, 5.25))
    
    title_ = ''
    
    plt.plot(matrixRobustness[:, 0], matrixRobustness[:, station],
             color = (.5, .5, .9, .50), label = "Robustez")
 
    title_ = 'Acurácia vs '
    plt.plot(tableMeanAccuracy[:, 0], tableMeanAccuracy[:, station],
         color = (.9, .0, .0, .50), label = "Acurácia")
    plt.title(title_ + 'Robustez  em função da dispersão\nentrada \"'
              + np.str(y_cols[station]) + '\"')
    plt.xlabel('Dispersão')    
    plt.axis(ymin = 0, ymax = 1.2)
    plt.legend(loc = 'upper center', bbox_to_anchor = (-0.30, 1),
          ncol = 1, fancybox = True, shadow = True, title = 'Legenda')
    plt.savefig(DIR + "2.0.2 acurácia vs robustez no em função do std "
                + np.str(y_cols[station]) + ' ' + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)    
    plt.show() 
    
    #3.3.2 acurácia no em função do std #######################################
    plt.close('all')
    #plt.figure(figsize = (12.5, 5.25))
    plt.title('Acurácia em função da dispersão\nentrada \"'
              + np.str(y_cols[station]) + '\"')
    plt.plot(tableMeanAccuracy[:, 0], tableMeanAccuracy[:, station],
             color = (.9, .0, .0, .50), label = "Acurácia")
    plt.xlabel('Dispersão')
    plt.ylabel('Acurácia')
    plt.legend(loc = 'upper center', bbox_to_anchor = (-0.30, 1),
          ncol = 1, fancybox = True, shadow = True, title = 'Legenda')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.savefig(DIR + "2.0.3 acurácia no em função do std "
                + np.str(y_cols[station])  + ' ' + typeOfCase +  ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show() 

    # 3.3.3 robustez no em função do std ######################################
    plt.close('all')
    #plt.figure(figsize = (12.5, 5.25))
    plt.title('Robustez  em função da dispersão\nentrada \"'
              + np.str(y_cols[station]) + '\"')
    plt.plot(matrixRobustness[:, 0], matrixRobustness[:, station],
             color = (.5, .5, .9, .50), label = "Robustez")
    plt.xlabel('Desvio padrão de entrada')
    plt.ylabel('Robustez')
    plt.legend(loc = 'upper center', bbox_to_anchor = (-0.30, 1),
          ncol = 1, fancybox = True, shadow = True, title = 'Legenda')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.savefig(DIR + "2.0.4 robustez no em função do std "
                + np.str(y_cols[station])  + ' ' + typeOfCase +  ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show() 
    
    


# Fim 2 - Análise estática variando std





###############################################################################
#
# 4 - Análise estática variando std
#
# Definir o desvio padrão desvio e executar
#
# 1 Plot para cada estação, std 1, 2 e 5
#
# Desvio 1 padrão  para todas as estações
#
###############################################################################

print("4 - Statis Analysis varying std")
# Para armazenar os dados
lineStd = []
lineAccuracy = []
lineRobustness = []
matrixStd = []
matrixAccuracy = []
matrixRobustness = []

plotHist = 1
# Tamanho das amostras
lenSamples = 100
genRandonNum = gauss()
# Definir desvios:
listStdDeviation = [0.1, 1, 2.0]
# Verifica os momentosEstatisticosSerie das entradas
moments = statsMomentsOfSeries(dataFrameInputData)
mon = moments[ : , MEAN]
for stdDeviation in listStdDeviation: # range(1, 11):# para cada desvio
    lineStd = [np.float64(stdDeviation)]
    lineAccuracy = [np.float64(stdDeviation)]    
    lineRobustness = [np.float64(stdDeviation)]

    for station in range(len(moments)): # Para cada estação
        mon, obsv, tt_ = maxSeriesInTimeT(dataFrameInputData, dataFrameObservedData, station) #usar para valores extremos
        arrayMom = np.array(mon)
        meanStation = arrayMom[station] # Pegar o valor da estação e usar como média da distribuição
        genRandonNum.setSeed(seed)
        arrayOfNoise = genRandonNum.gaussiana(meanStation, stdDeviation, lenSamples) # Gera ruído centrado no valor extremo da estação
        arrayInputDataWithNoise = []
    
        for noiseIndex in range(lenSamples): # Para cada ruido gerado
                arrayMom[station] = arrayOfNoise[noiseIndex]
                arrayInputDataWithNoise.append(arrayMom.copy()) # Monta linha a linha
                
        arrayInputDataWithNoise = np.array(arrayInputDataWithNoise)
        arrayInputDataWithNoiseNorm = scalerInput.transform(arrayInputDataWithNoise) # Normaliza os dados
        
        # Previsão
        predict_, norm_prediction = predicao_(scalerInput, scalerObserved, model, arrayInputDataWithNoiseNorm) # Passa os dados pelo modelo e pega a previsão
        
        lineStd.extend([np.std(predict_[: , FORECAST_STATION])]) # Pega o desvio da previsão (conselheiro paulino)

        lineRobustness.extend([robustness(arrayInputDataWithNoise[: , station],
                                      predict_[: , FORECAST_STATION])]) # Calcula a robustez e armazena
        
        lineAccuracy.extend([accuracyOfMean([obsv],
                                    [np.mean(predict_[: , FORECAST_STATION])])]) #conselheiro paulino
        if(plotHist == 1):
            # 2.0.3 histograma das estações #############################################  
            plt.close('all')   
            plt.figure(figsize = (10.5, 5.25))
            #plt.title('Distribuição em função da desvio de estacao')
            plt.subplot(121)
            plt.title('Estação \"' + np.str(inputNames[station])
                      + '\" desvio = ' + np.str(stdDeviation))
            
            n1, bins1, patches1 = plt.hist(arrayOfNoise, bins = 'auto',
                                 histtype = 'stepfilled',
                                 facecolor = 'lime')    
            inputMean = np.average(arrayOfNoise)
            inputMedian = np.median(arrayOfNoise)
            #print(np.where(n1 == np.amax(n1)))
            jj = np.where(n1 == np.amax(n1))
            #print(j)
            ii = np.int(np.double(jj[0][0]))    
            inputMode = (bins1[ii] + bins1[ii + 1]) / 2
            plt.axvline(inputMean, color = 'blue', linewidth = 2,label = 'Média [' +"{:.2f}".format(inputMean) +']', alpha = 0.5)
            plt.axvline(inputMode, color = 'red', linewidth = 2, label = 'Moda [' +"{:.2f}".format(inputMode) +']', alpha = 0.5)
            plt.axvline(inputMedian, color = 'green', linewidth = 2, label = 'Mediana [' +"{:.2f}".format(inputMedian) +']', alpha = 0.5)
            plt.xlabel(listStationsMetrics[station])
            plt.ylabel('Frequência')        
            plt.xticks(rotation=45)
            plt.legend(loc = 'best')   
            
            
            plt.subplot(122)
            plt.title('Previsão Nível \"Conselheiro\nPaulino\" desvio = '
                      + np.str(np.std(predict_[: , FORECAST_STATION])))
            
            # Caso o desvio padrão seja igual a zero, ajusta o plot
            if(np.std(predict_[: , FORECAST_STATION]) < 0.001):
                xmin_ = np.mean(predict_[: , FORECAST_STATION]) - 3
                xmax_ = np.mean(predict_[: , FORECAST_STATION]) + 3
                bins1 = np.linspace(xmin_, xmax_, num = len(bins1))
               
            n2, bins2, patches2 = plt.hist(predict_[: , FORECAST_STATION],
                                           bins = 'auto', histtype = 'stepfilled',
                                           facecolor = 'lime')
            meanPrediction = np.average(predict_[: , FORECAST_STATION])
            medianPrediction = np.median(predict_[: , FORECAST_STATION])
            jj = np.where(n2 == np.amax(n2))      
            ii = np.int(np.double(jj[0][0]))    
           
            modePrediction = (bins2[ii] + bins2[ii + 1]) / 2
            plt.axvline(meanPrediction, color = 'blue', linewidth = 2,label = 'Média [' +"{:.2f}".format(meanPrediction) +']', alpha = 0.5)
            plt.axvline(modePrediction, color = 'red', linewidth = 2, label = 'Moda [' +"{:.2}".format(modePrediction) +']', alpha = 0.5)
            plt.axvline(medianPrediction, color = 'green', linewidth = 2, label = 'Mediana [' +"{:.2f}".format(medianPrediction) +']', alpha = 0.5)
            plt.axvline(obsv, color = 'black', linewidth = 1, label = 'Observado [' +"{:.2f}".format(obsv) +']', linestyle = '--')
            plt.xlabel('Nível [m]')
            plt.ylabel('Frequência')        
            plt.xticks(rotation=45)
            plt.legend(loc = 'best')
            plt.savefig(np.str(DIR + "4.0 histograma das estações "
                               + np.str(inputNames[station]) + " desvio " 
                               + np.str(stdDeviation) + " " + typeOfCase
                               + ".png"), bbox_inches = 'tight',
                        dpi = imageResolution)
            plt.show()
        
    matrixStd.append(np.array(lineStd))    
    matrixAccuracy.append(np.array(lineAccuracy))
    matrixRobustness.append(np.array(lineRobustness))
    

matrixStd = np.array(matrixStd)
matrixAccuracy = np.array(matrixAccuracy)   
matrixRobustness = np.array(matrixRobustness) 
  
tableStds = pd.DataFrame(data = matrixStd)
col_s = ['Std']
col_s.extend(inputNames)
tableStds.columns = col_s

tableStds.to_csv (DIR + '4 Análise estática variando std ' + typeOfCase + '.csv',
                    index = False, header = True)

matrixAccuracy = pd.DataFrame(data = matrixAccuracy)
col_s = ['Accuracy']
col_s.extend(inputNames)
matrixAccuracy.columns = col_s

matrixAccuracy.to_csv (DIR + '4 Análise estática variando std acuracia ' + typeOfCase + '.csv',
                    index = False, header = True)

matrixRobustness = pd.DataFrame(data = matrixRobustness)
col_s = ['Robstness']
col_s.extend(inputNames)
matrixRobustness.columns = col_s

matrixRobustness.to_csv (DIR + '4 Análise estática variando std robustez '  + typeOfCase + '.csv',
                    index = False, header = True)



#######################################
# 4.1 Bargraph for std output and input
#
plt.close('all')
x_bar = np.arange(len(tableStds.iloc[0 , 1:]))
fig, ax = plt.subplots()
plt.title('Dispersão na previsão em relação\nao desvio padrão nas estações '  + typeOfCase)
lenSamplesStd = len(tableStds.iloc[:, 0])
for stdDeviation in range(lenSamplesStd):
    plt.bar(x_bar + stdDeviation / (lenSamplesStd + 1), tableStds.iloc[stdDeviation , 1:],
            align = 'center', width=1 / (lenSamplesStd + 1),
            label = str(tableStds.iloc[stdDeviation , 0]))
plt.xticks(x_bar, col_s[1:], rotation = 45, horizontalalignment = 'right')
plt.axis(ymin = 1.E-06, ymax = 10)
ax.set_yscale('log')
#plt.xlabel('Estações')
plt.ylabel('Desvio Padrão previsão [log]')
plt.legend(loc = 'upper right', bbox_to_anchor = (-0.2, 1), ncol = 1,
               fancybox = True, shadow = True, title =  'Desvio padrão\n  na estação')
fig.savefig(DIR + "4.1 - Previsão de nível para perturbação em cada estacao (log) "
             + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()


#######################################
# 4.2 Bargraph for accuracy vs std input
#
plt.close('all')
x_bar = np.arange(len(tableStds.iloc[0 , 1:]))
fig, ax = plt.subplots()
plt.title('Acurácia em relação\nao desvio padrão nas estações '   + typeOfCase)
lenSamplesStd = len(tableStds.iloc[:, 0])
for stdDeviation in range(lenSamplesStd):
    plt.bar(x_bar + stdDeviation / (lenSamplesStd + 1), matrixAccuracy.iloc[stdDeviation , 1:],
            align = 'center', width=1 / (lenSamplesStd + 1),
            label = str(tableStds.iloc[stdDeviation , 0]))
plt.xticks(x_bar, col_s[1:], rotation = 45, horizontalalignment = 'right')
plt.axis(ymin = 0, ymax = 1)
#ax.set_yscale('log')
#plt.xlabel('Estações')
plt.ylabel('Desvio Padrão previsão [log]')
plt.legend(loc = 'upper right', bbox_to_anchor = (-0.2, 1), ncol = 1,
               fancybox = True, shadow = True, title =  'Acurácia\n  na estação')
fig.savefig(DIR + "4.2 - Acurácia para perturbação em cada estacao (log) "
             + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()



#######################################
# 4.3 Bargraph for robustness vs std input
#
plt.close('all')
x_bar = np.arange(len(tableStds.iloc[0 , 1:]))
fig, ax = plt.subplots()
plt.title('Robustez em relação\nao desvio padrão nas estações '  + typeOfCase)
lenSamplesStd = len(tableStds.iloc[:, 0])
for stdDeviation in range(lenSamplesStd):
    plt.bar(x_bar + stdDeviation / (lenSamplesStd + 1), matrixRobustness.iloc[stdDeviation , 1:],
            align = 'center', width=1 / (lenSamplesStd + 1),
            label = str(tableStds.iloc[stdDeviation , 0]))
plt.xticks(x_bar, col_s[1:], rotation = 45, horizontalalignment = 'right')
plt.axis(ymin = 0, ymax = 1)
#ax.set_yscale('log')
#plt.xlabel('Estações')
plt.ylabel('Desvio Padrão previsão [log]')
plt.legend(loc = 'upper right', bbox_to_anchor = (-0.2, 1), ncol = 1,
               fancybox = True, shadow = True, title =  'Robustez\n  na estação')
fig.savefig(DIR + "4.3 - Robustez para perturbação em cada estacao (log) "
             + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
plt.show()





# # O que era para plotar aqui?
# plt.close('all')
# fig, ax = plt.subplots()
# ax.set(xlabel = 'Desvio Padrão estação', ylabel = 'Robustez', title = '')
# for station in range(1, len(tableStds.iloc[0, :])):
#     ax.plot(matrixRobustness[: , 0], matrixRobustness[: , station], label = y_cols[station])
# ax.grid()
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend(loc = 'upper center', bbox_to_anchor = (-0.5, 1),
#           ncol = 1, fancybox = True, shadow = True,
#           title = 'Previsão de nível para\nperturbação em cada estação')

# fig.savefig("robustez.png",
#             bbox_inches = 'tight', dpi = imageResolution)
# plt.show()

# Fim 4 - Análise estática variando std


###############################################################################
#
# 5 - Dinâmica
#
###############################################################################
print("5 - Synchronous Analysis fixed std")
station  = 0
stdDeviation = 0.1 #Desvio fixo para cada estação de estacao
observed = arrayObservedData[ : , FORECAST_STATION]


arrayInputDataRawLag = arrayInputDataRaw[LAG_MIN : ]

#moments = statsMomentsOfSeries(dataFrameInputData)

# Itera por entradas
while(station < len(inputNames)):           
    #tamanho das amostras
    lenSamples = 100
    genRandonNum = gauss()
    limits = []
    arrayLimits = []
    arrayRobustness = []
    #l = 1
    
    # Tamanho linhas e colunas
    nlin = len(arrayNormInputs)
    ncol = len(arrayNormInputs[0])
    
    scat_x = []
    scat_y = []
    
    #nlin = 100
    max_prev = 0
    
    print(inputNames[station])  
    arrayAccuracy = []
    for t in range(0, nlin): # Varia no tempo t (timestamp)
        arrayMom = np.copy(arrayDenormInputs[t])# Pega uma linha no dataset no tempo t
        meanStation = arrayMom[station] # Pega o valor da estacao no tempo t
        genRandonNum.setSeed(seed)
        arrayOfNoise = genRandonNum.gaussiana(meanStation, stdDeviation, lenSamples) # Gera ruido aleatório
        arrayInputDataWithNoise = [] # Monta estrutura de dados para usar no modelo
        
        arrayInputDataWithNoise.append(arrayMom.copy()) # Monta 1a linha sem ruido
        
        for k in range(lenSamples): # Para cada ruido gerado
                arrayMom[station] = arrayOfNoise[k] # Atribui para cada posição da estacao o ruído
                arrayInputDataWithNoise.append(arrayMom.copy()) # Monta linha a linha

        arrayInputDataWithNoise = np.array(arrayInputDataWithNoise)
        arrayInputDataWithNoiseNorm = scalerInput.transform(arrayInputDataWithNoise) # Normaliza os dados        
        predict_, norm_prediction = predicao_(scalerInput, scalerObserved, model,
                                              arrayInputDataWithNoiseNorm) # Passa os dados pelo modelo treinado
            
        #matrixStd.extend([np.std(predict_[: , FORECAST_STATION])])
        prediction = np.copy(predict_[: , FORECAST_STATION]) # Saída (Conselheiro Paulino)
        # Atribui momentosEstatisticosSerie estatísticos:
        med = np.mean(prediction)
        stdv = np.std(prediction)
        min_ = min(prediction)
        max_ = max(prediction)
        limits.extend([min_, (med - stdv), med, (med + stdv), max_]) # Monta a linha de momentosEstatisticosSerie
        arrayLimits.append(limits) # Insere a linha na matriz de momentosEstatisticosSerie (cada linha representa t)
        limits = []
        obsv = observed[t] # Pega o dado observed
   
        arrayAccuracy.append(accuracyOfMean([obsv],
                            [np.mean(predict_[: , FORECAST_STATION])])) # Monta matriz acurácia (cada linha é t)
        arrayRobustness.append(robustness(arrayInputDataWithNoise[: , station], prediction))  # Monta matriz robustez (cada linha é t)
        
        #dados para plotagem
        scat_x.append(med) # Média
        scat_y.append(stdv) # Desvio Padrão
        if(max(prediction) > max_prev):
            max_prev = max(prediction)
            listInputNoise = []
            listPredictionDisturbance = []
            listInputNoise = np.copy(arrayOfNoise)
            listPredictionDisturbance = np.copy(prediction)
            sts_ = [min_, max_, med, stdv]   # sem uso ainda 
        
    if(t == 100):
        print("sai")
    arrayLimits = np.array(arrayLimits)
    timestamp = np.arange(0, len(arrayLimits), 1) # Gera o timestamp do tamanho das linhas da matriz
    '''
    t = 0
    xBoxplot = []
    arrayBoxplot = []
    while(t < nlin):
        xBoxplot.append(timestamp[t])
        arrayBoxplot.append(np.array(arrayLimits[t]) )
        t = int(t + nlin / 4)       
    '''
    
    
    print("5.0 Plot da incerteza")
    # 5.0 Plot da incerteza####################################################
    #arrayBoxplot = np.array(arrayBoxplot)
    #xBoxplot = np.array(xBoxplot)
    colorStd = (.5, .0, .0, 1)
    colorMean = (.5, .0, .0, 1)
    colorExtrems = (.5, .0, .0, 1)
    colorStdArea = (.5, .5, .5, .5)       
    
    plt.close('all')
    plt.figure(figsize = (10.5, 5.25))
    plt.title('Entrada \"' + np.str(inputNames[station]) + '\" dispersão ruído = '
              + np.str(stdDeviation))
    plt.plot(timestamp, observed, color = 'green', label = "Observado",
             linewidth = 0.5)
    plt.plot(timestamp, arrayLimits[:, 2], label = "Média simulação", color = colorMean,
             linewidth = 0.5)
    plt.fill_between(timestamp, arrayLimits[:, 1], arrayLimits[:, 3], color = colorStdArea,
                     label = "Desvio padrão") # area colorida entre desvios
    plt.plot(timestamp, arrayLimits[:, 0], color = 'blue', label = "Mínimo simulação",
             linewidth = 0.5)
    plt.plot(timestamp, arrayLimits[:, 4], color = 'red', label = "Máximo simulação",
             linewidth = 0.5)
    #plt.boxplot(arrayBoxplot, positions = xBoxplot, widths = 1000)
    #plt.plot(timestamp, arrayLimits[:, 1], timestamp, arrayLimits[:, 3], color = colorStd, linestyle = 'dotted', label = "Desvio padrão")
    plt.legend(loc = 'upper center', bbox_to_anchor = (-0.20, 1), ncol = 1,
               fancybox = True, shadow = True, title = 'Legenda')
    plt.ylabel('Nível [m]')
    plt.xlabel('Timestamp')
    plt.savefig(DIR + "5.0 - incerteza na previsão "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()    
    
    print("5.1 - scatter incerteza vs magnitude")
    # 5.1 - scatter incerteza vs magnitude#####################################
    plt.close('all')
    plt.figure(figsize = (10.5, 5.25))
    plt.title('Entrada \"' + np.str(inputNames[station]) + 
              '\" desvio padrão do ruído = ' + np.str(stdDeviation))
    #plt.plot(arrayLimits[:, 2], arrayLimits[:, 3] - arrayLimits[:, 2], 'o',
    #         color = (.5, .5, .9, .1))
    
    plt.scatter(arrayLimits[:, 2], arrayLimits[:, 3] - arrayLimits[:, 2],
             s=80, facecolors='none', edgecolors= (0, 0, 1, .5))
    plt.xlabel('Média')
    plt.ylabel('Desvio padrão')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.savefig(DIR + "5.1 - scatter desvio padrão vs média "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) +
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    '''
    print("5.2 histograma das estações")
    # 5.2 histograma das estações #############################################  
    plt.close('all')   
    plt.figure(figsize = (10.5, 5.25))
    plt.subplot(121)
    plt.title('Estação \"' + np.str(inputNames[station]) + '\" dispersão ruído = '
              + np.str(stdDeviation))
    n, _y, patches1 = plt.hist(listInputNoise, bins = 'auto', histtype = 'stepfilled',
                         facecolor = 'lime')    
    inputMean = np.average(listInputNoise)
    inputMedian = np.median(listInputNoise)
    #print(np.where(n == np.amax(n)))
    j = np.where(n == np.amax(n))
    #print(j)
    i = np.int(np.double(j[0][0]))    
    inputMode = (_y[i] + _y[i + 1]) / 2
    plt.axvline(inputMean, color = 'blue', linewidth = 3,label = 'Média')
    plt.axvline(inputMode, color = 'red', linewidth = 3, label = 'Moda')
    plt.axvline(inputMedian, color = 'green', linewidth = 3, label = 'Mediana')
    plt.xlabel(listStationsMetrics[station])
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.legend(loc = 'upper right')       
    plt.subplot(122)
    plt.title('Dispersão na previsão')
    n_, y_, patches2 = plt.hist(listPredictionDisturbance, bins = 'auto',
                                histtype = 'stepfilled', facecolor = 'lime')
    meanPrediction = np.average(listPredictionDisturbance)
    medianPrediction = np.median(listPredictionDisturbance)
    #print(np.where(n == np.amax(n)))
    j_ = np.where(n_ == np.amax(n_))
    #print(j)
    ii = np.int(np.double(j_[0][0]))    
    modePrediction = (y_[ii] + y_[ii + 1]) / 2
    plt.axvline(meanPrediction, color = 'blue', linewidth = 3,label = 'Média')
    plt.axvline(modePrediction, color = 'red', linewidth = 3, label = 'Moda')
    plt.axvline(medianPrediction, color = 'green', linewidth = 3, label = 'Mediana')
    plt.xlabel('Nível [m]')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.legend(loc = 'upper right')
    plt.savefig(DIR + "5.2 histograma das estações "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + ".png",, format = 'png',
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    print("5.3 Plot da media vs incerteza")
    # 5.3 Plot da media vs incerteza ##########################################
    plt.close('all')
    plt.figure(figsize = (10.5, 5.25))
    plt.title('Média vs Incerteza para estação \"'
              + np.str(inputNames[station]) + '\" dispersão ruído = '
              + np.str(stdDeviation))
    plt.plot(arrayLimits[:, 2], arrayLimits[:, 3] - arrayLimits[:, 2],
             'o', color = (.5, .5, .9, .1))
    plt.xlabel('Média')
    plt.ylabel('Incerteza')
    plt.axis(ymin = 0, ymax = 1.2, xmin = 0, xmax = 1.2)
    plt.savefig(DIR + "5.3 Plot da media vs incerteza "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
                bbox_inches = 'tight', dpi = imageResolution)    
    plt.show()
    '''
    print("5.4 Plot da acurácia previsão no tempo")
    # 5.4 Plot da acurácia previsão no tempo ##################################
    plt.close('all')     
    fig, ax_left = plt.subplots(figsize = (10.5, 7.0))
    ax_right = ax_left.twinx()

    p1, = ax_left.plot(timestamp, arrayAccuracy, color = 'orange', label = "Acurácia", linewidth = 0.5)
    p2, = ax_right.plot(timestamp, observed, color = 'green', label = "Nível [m]", linewidth = 0.5)
    
    ax_left.set_xlabel('Timestamp')
    ax_left.set_ylabel('Acurácia')        
    ax_right.set_ylabel('Observado')
    ax_left.yaxis.label.set_color(p1.get_color())
    ax_right.yaxis.label.set_color(p2.get_color())
    lns = [p1, p2]
    ax_left.legend(handles=lns, loc='best')
    ax_left.axis(ymin = 0, ymax = 2)    
    ax_right.axis(ymin = -2, ymax = 4)
    plt.title('Acurácia para estação \"' + np.str(inputNames[station]) 
              + '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.savefig(DIR + "5.4 Plot da acurácia previsão no tempo "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
     
    print("5.5 Plot da acurácia vs robustez")
    # 5.5 Plot da acurácia vs robustez no tempo ###############################
    plt.close('all')
    plt.title('Acurácia vs Robustez para estação \"'
              + np.str(inputNames[station]) +
              '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.scatter(arrayRobustness, arrayAccuracy, 
             s=80, facecolors='none', edgecolors= (0, 0, 1, .5))
    plt.xlabel('Robustez')
    plt.ylabel('Acurácia')
    plt.axis(ymin = 0, ymax = 1.2, xmin = 0, xmax = 1.2)
    plt.savefig(DIR + "5.5 Plot da acurácia vs robustez no tempo "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    #acuracia_total = 1 - np.sum(arrayAccuracy) / (len(arrayAccuracy))    
    
    print("5.6 Plot da acurácia vs observado")
    # 5.6 Plot da acurácia vs observado no tempo ##############################
    plt.close('all')
    plt.figure(figsize = (12.5, 5.25))
    plt.title('Acurácia vs observado para estação \"'
              + np.str(inputNames[station]) +
              '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.scatter(observed, arrayAccuracy,
             s=80, facecolors='none', edgecolors= (0, 0, 1, .5))
    plt.xlabel('Observado')
    plt.ylabel('Acurácia')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.savefig(DIR + "5.6 Plot da acurácia vs observado "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    print("5.7 Plot da robustez previsão no tempo")
    # 5.7 Plot da robustez previsão no tempo ##################################
    plt.close('all')   
    fig, ax_left = plt.subplots(figsize = (10.5, 7.0))
    ax_right = ax_left.twinx()

    p1, = ax_left.plot(timestamp, arrayRobustness, color = 'orange', label = "Robustez", linewidth = 0.5)
    p2, = ax_right.plot(timestamp, observed, color = 'green', label = "Nível [m]", linewidth = 0.5)
    
    ax_left.set_xlabel('Timestamp')
    ax_left.set_ylabel('Robustez')        
    ax_right.set_ylabel('Observado')
    ax_left.yaxis.label.set_color(p1.get_color())
    ax_right.yaxis.label.set_color(p2.get_color())
    lns = [p1, p2]
    ax_left.legend(handles=lns, loc='best')
    ax_left.axis(ymin = -.5, ymax = 2)    
    ax_right.axis(ymin = -2, ymax = 4)
    plt.title('Robustez para estação \"'+ np.str(inputNames[station]) 
              + '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.savefig(DIR + "5.7 Plot da robustez previsão no tempo "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()

    print("5.8 Plot da robustez vs observado")
    # 5.8 Plot da robustez vs observado no tempo ##############################
    plt.close('all')
    plt.figure(figsize = (12.5, 5.25))
    plt.title('Robustez vs observado para estação \"'
              + np.str(inputNames[station]) +
              '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.scatter(observed, arrayRobustness,
             s=80, facecolors='none', edgecolors= (0, 0, 1, .5))
    plt.xlabel('Observado')
    plt.ylabel('Robustez')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.savefig(DIR + "5.8 Plot da robustez vs observado "
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    print("5.9 Plot da Acurácia e Robustez vs Observado exultorio")
    # 5.9 Plot da Acurácia e Robustez vs robustez no tempo ###############################
    plt.close('all')
    plt.figure(figsize = (12.5, 5.25))
    plt.title('Acurácia e Robustez vs Observado para estação \"'
              + np.str(inputNames[station]) +
              '\" dispersão ruído = ' + np.str(stdDeviation))
    plt.scatter(observed, arrayAccuracy, s=80, facecolors='none', edgecolors= (0, 0, 1, .5), label = 'Acurácia')
    plt.scatter(observed, arrayRobustness, marker = 'v', 
                s=80, facecolors='none', edgecolors= (0, 1, 0, .5), label = 'Robustez')
    plt.xlabel('Observado')
    # plt.ylabel('Métrica')
    plt.axis(ymin = 0, ymax = 1.2)
    plt.legend(loc = 'best')
    plt.savefig(DIR + "5.9 Plot da Acurácia e Robustez vs Observado exultorio"
                + np.str(inputNames[station]) + " std " + str(stdDeviation) + 
                " " + typeOfCase + ".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()


    # Itera para próxima estacao
    station  = station + 1
    

# Fim 5 - Dinâmica



###############################################################################
#
# 7 - Controle do desvio padrão para 10% de valoes negativos
#
#
###############################################################################
print("7 - Negative values calc")
# Usar para valores extremos de todas as esta~ções quando NCP estiver no max
mon, obsv, tt_ = maxSeriesInTimeT(dataFrameInputData, dataFrameObservedData, INTEREST_STATION) #usar para valores extremos
             
listStdsForStatioInMaxOutput = []  
listStdsForMaxInStation = [] 
listStdsForMeanInStation = []  

for station in range(len(moments)):
    # Max no observado
    arrayMom = np.array(mon)
    maxStationCP = arrayMom[station]
    # genRandonNum.setSeed(seed)
    # arrayOfNoise = genRandonNum.gaussiana(maxStationCP, stdDeviation, lenSamples)
  
    #print(maxStationCP)
    stdMax = limStd(0, 100, 0.1, genRandonNum, seed, maxStationCP, lenSamples, 0.1)  
    listStdsForStatioInMaxOutput.append(stdMax)
    
    # Media de cada estacao
    meanStation = np.mean(arrayInputData[ : , station])
    #print(meanStation)
    stdMeanStation = limStd(0, 100, 0.1, genRandonNum, seed, meanStation, lenSamples, 0.1)  
    listStdsForMeanInStation.append(stdMeanStation)
    
    # Max de cada estação
    maxStationStd = max(arrayInputData[ : , station])
    #print(maxStationStd)
    stdMaxStation = limStd(0, 100, 0.1, genRandonNum, seed, maxStationStd, lenSamples, 0.1)  
    listStdsForMaxInStation.append(stdMaxStation)
    


matrixTable = np.array(listStdsForStatioInMaxOutput) 
matrixTable = pd.DataFrame(data = np.transpose(matrixTable))
matrixTable.columns = inputLabels
matrixTable.to_csv (DIR + ' 7 Std das estcoes para nivel max CP ' + typeOfCase + '.csv',
                    index = False, header = True)


matrixTable = np.array(listStdsForMeanInStation) 
matrixTable = pd.DataFrame(data =np.transpose(matrixTable))
matrixTable.columns = inputLabels
matrixTable.to_csv (DIR + ' 7 Std das estcoes para nivel medio em cada estacao '
                    + typeOfCase + '.csv',
                    index = False, header = True)


matrixTable = np.array(listStdsForMaxInStation) 
matrixTable = pd.DataFrame(data = np.transpose(matrixTable))
matrixTable.columns = inputLabels
matrixTable.to_csv (DIR + ' 7 Std das estacoes para nivel max em cada estacao '
                    + typeOfCase + '.csv',
                    index = False, header = True)


