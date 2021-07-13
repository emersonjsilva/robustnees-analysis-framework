#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#
# Modelo hidrológico
#
###############################################################################

import pandas as pd
import numpy as np

#from keras.datasets import mnist
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.models import Sequential
from keras.layers import Reshape, Conv2DTranspose, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import max_norm
from keras.constraints import unit_norm

import matplotlib.pyplot as plt
import datetime
from metrics import *


##############################################################################
#                           Parâmetros do modelo
# Fixed values
FORECAST_STATION = 0 # Nível Concelheiro paulinho apenas para previsão!!!!!

# Each 15 minutes forecast lag_min = 1 Each 120 minutes forecast lag_min = 8
LAG_MIN = 1

# Nível Concelheiro paulinho medição exutório
INTEREST_STATION = 9 #

# Pasta de saida dos graficos e tabelas
DIR = 'Anexos_' + str(LAG_MIN * 15) +'_min/' #Saida dos arquivos

# Resolução dos graficos
imageResolution = 100

# Épocas de treinamento
epochs_ = 100

# Batchs
batchs_ = 10
##############################################################################

listaNSE = []
listaRMSE = []

# Carrega dataset em CSV
def readCsvDataset(data_path, source):
  df = pd.read_csv('{}/{}.csv'.format(data_path, source), parse_dates=['data_hora'])
  df = df.set_index('data_hora')
  #df.to_csv("dataset_.csv")
  return df

# Carrega dataset em XLS 
def read_xls_dataset(data_path, source):
  df = pd.concat(pd.read_excel('{}/{}.xlsx'.format(data_path, source), 
                                sheet_name=None), ignore_index=True)
  df.columns = ['data_hora', 'chuva_{}'.format(source), 'nivel_{}'.format(source)]
  df = df.set_index('data_hora')
  return df

# Retorna valor máximo
def get_max(validation, txt):
    #validation = history.history["val_accuracy"]
    ymax = max(validation)
    return "Max " + txt + " ≈ " + "%.2f" % ymax + "%"

# Plots treinamentos
def plot_history(history, title):
  print(history.history.keys())
  #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 6))
  
  plt.close('all')   
  plt.figure(figsize=(10.5, 5.25))
  plt.title("Nash-Sutcliffe Efficiency - treinamento") 
  
  plt.title("Treinamento - Nash-Sutcliffe Efficiency Max ≈ " + np.str(max(history.history['val_nse'])))
  plt.plot(history.history['nse'], label='train')
  plt.plot(history.history['val_nse'], label='valid')
  plt.ylabel('Nash-Sutcliffe Efficiency')
  plt.xlabel('Épocas')
  plt.legend()  
  plt.savefig(DIR + "0.0 - Treinamento nse.png",
            bbox_inches = 'tight', dpi = imageResolution)
  plt.show()  
  
  plt.close('all')   
  plt.figure(figsize=(10.5, 5.25))
  plt.title("Treinamento - Loss Min ≈ " + np.str(min(history.history['val_loss'])) )
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='valid')
  plt.ylabel('Loss')
  plt.xlabel('Épocas')
  plt.legend()  
  plt.savefig(DIR + "0.0 - Treinamento loss.png",
            bbox_inches = 'tight', dpi = imageResolution)
  plt.show()

  plt.close('all')   
  plt.figure(figsize=(10.5, 5.25))
  plt.title("Treinamento - Root Mean Square Error Max ≈ " + np.str(max(history.history['val_rmse'])) )
  plt.plot(history.history['rmse'], label='train')
  plt.plot(history.history['val_rmse'], label='valid')
  plt.ylabel('Root Mean Square Error')
  plt.xlabel('Épocas')
  plt.legend()  
  plt.savefig(DIR + "0.0 - Treinamento rmse.png",
            bbox_inches = 'tight', dpi = imageResolution)
  plt.show()    
  
  trainning_metrics = pd.DataFrame(np.transpose([ history.history['loss'],
                                                 history.history['val_loss'],
                                                  history.history['nse'],
                                                  history.history['val_nse'],
                                                  history.history['rmse'],
                                                  history.history['val_rmse']
                                                 ]))
  
  trainning_metrics.columns = ['loss', 'val_loss', 'nse',
                               'val_nse', 'rmse', 'val_rmse']
  
  trainning_metrics.to_csv(DIR + "trainning_metrics.csv")  

# Métrica Nash Sutcliffe Error para treinamento    
def nse(y_true, y_pred):
  listaNSE.append([y_true, y_pred])
  return 1 - (K.sum((y_pred - y_true)**2)/K.sum((y_true - K.mean(y_true))**2))

# Métrica Root Mean Squared Error
def rmse(y_true, y_pred):    
    #listaRMSE.append([y_true, y_pred])
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

# Modelo neural
#def hydrologicalModel (treinarRede, carregarPesosRede, data, plots_):
def hydrologicalModel (treinarRede, carregarPesosRede, dadosEntrada, dadosObservado, plots_):
    ###########################################################################
    #
    # Carrega dataset
    #
    ###########################################################################

    # Normaliza entre 0 e 1 com 10% de tolerancia inferior e superior
    scalerx = MinMaxScaler(feature_range=(0.1, 0.9))
    scalery = MinMaxScaler(feature_range=(0.1, 0.9))

    # Normaliza normaliza dataset entrada e observado
    x_ = scalerx.fit_transform(dadosEntrada)
    y_ = scalery.fit_transform(dadosObservado)
    
    # Divide o numero de amostras na metade
    len_x_train = np.int32(np.around(len(x_)*0.5))
    len_y_train = np.int32(np.around(len(y_)*0.5))    
    # Separa treino 50% e testes 50% (para calibrar modelo)
    x_train_, x_test, y_train_, y_test = x_[0:len_x_train], x_[len_x_train+1:], y_[0:len_x_train], y_[len_x_train+1:]
    
    # Separa 20% do treino para validação cruzada
    x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=0.2)
    

    #print(x_train.shape, y_train.shape)
    #print(x_test.shape, y_test.shape)
    
    ###########################################################################
    #
    # Configura rede
    #
    ###########################################################################
    
    encoded_dim = 5
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    #print(input_dim)
    
    # Arquivo com os pesos
    checkpointer_file = 'checkpoint_model_' + str(LAG_MIN * 15) +'_min.hdf5'
    
    # Configuração do modelo
    model = Sequential()  
    
    '''
    model.add(Dense(input_dim,
                    input_shape=(input_dim,),
                    activation='linear'))
    model.add(Dense(5,
                    activation='linear'))
    model.add(Dense(output_dim,activation='relu',
                    activity_regularizer=l2(0.0001)))
    '''


    model.add(Dense(5,input_shape=(input_dim,),activation='relu'))
    model.add(Dense(1, activation='linear'))

    # seed(seed_number)
    # tf.random.set_seed(seed_number)
    # np.random.seed(seed_number)
    # Carregar Pesos
    if carregarPesosRede == 1:
        model.load_weights(checkpointer_file)
        
    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[rmse, nse])
    '''
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=[rmse, nse]) #metrics=['accuracy'])
    '''
    if(plots_ == 1):
        model.summary()
        #print("summary")
    
    # Carrega pesos da rede
    checkpointer = ModelCheckpoint(filepath=checkpointer_file,
                                   verbose=1, 
                                   save_best_only=True)
    
    ###########################################################################
    #
    # Treinar
    #
    ###########################################################################
    
    if treinarRede == 1:
        # Treinamento
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs_,
                            batch_size=batchs_,
                            validation_data=(x_val, y_val),
                            callbacks=[checkpointer])
        #if(plots_ == 1):
        #print(history)
        print(listaNSE)
        plot_history(history, 'Treinamento')
        print("plot treinamento")
        
    ###########################################################################
    #
    # Predição
    #
    ###########################################################################
    
    predicao(scalerx, scalery, model,  x_test, y_test, plots_, 0, "Previsão " 
             + str(LAG_MIN * 15) +" minutos dado de testes")
    return x_, y_, scalerx, scalery, model

# Predição com plots 
def predicao(scalerx, scalery, model, estacoes, observado_, plots_, separador, titulo):    
    previsao = scalery.inverse_transform(model.predict(estacoes))
    observado = scalery.inverse_transform(observado_)
    #print(previsao.shape)
    
    if(plots_ == 0):
        return observado, previsao
    
    n_size = len(observado[:,0])
    
    
    # Previsão todo o dado observado vs previsão******************************
    plt.close('all')
    fig, (ax1) = plt.subplots(1,1, figsize=(20, 6))
    plt.plot(range(n_size),
             observado[:,FORECAST_STATION][:n_size],
             label='observado',
             color='blue')
    plt.plot(range(n_size),
             previsao[:,FORECAST_STATION][:n_size],
             label='previsto',
             color='orange',
             alpha=0.75)
    plt.legend()    
    plt.suptitle('Previsão de nível Conselheiro Paulino - ' + titulo)
    if(separador != 0):
        ax1.axvline(separador, color='black', linewidth=1, label='separador')
        ax1.text(
            (separador-(separador/2))/n_size,
            0.95,
            "Dados de treinamento e validação",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax1.transAxes,
            fontsize=14,
            )        
        ax1.text(
            (n_size-(n_size-separador)/2)/n_size,
            0.95,
            "Dados de testes",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax1.transAxes,
            fontsize=14,
            )        
    ax1.set_ylabel('Nível [m]')
    ax1.set_xlabel('Timestamp')
    plt.savefig(DIR + "0.0 - Previsão todo o dado observado vs previsão"+titulo+".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    
    # Plot do observado somente **********************************************
    plt.close('all')
    fig, (ax1) = plt.subplots(1,1, figsize=(20, 6))
    plt.plot(range(n_size),
             observado[:,FORECAST_STATION][:n_size],
             label='observado',
             color='blue')
    plt.legend()    
    plt.suptitle('Previsão de nível Conselheiro Paulino - Observado ' + titulo)
    ax1.set_ylabel('Nível [m]')
    ax1.set_xlabel('Timestamp')
    plt.savefig(DIR + "0.0 - Observado "+titulo+".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()    
    
    
    # Plot da previsão somente ***********************************************
    plt.close('all')
    fig, (ax1) = plt.subplots(1,1, figsize=(20, 6))
    plt.plot(range(n_size),
             previsao[:,FORECAST_STATION][:n_size],
             label='previsto',
             color='orange'
             )
    plt.legend()    
    plt.suptitle('Previsão de nível Conselheiro Paulino - Previsão ' + titulo)
    ax1.set_ylabel('Nível [m]')
    ax1.set_xlabel('Timestamp')
    plt.savefig(DIR + "0.0 - Previsão "+titulo+".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    # Scatter plot da previsão somente ***********************************************
    plt.close('all')
    plt.figure(figsize = (10.5, 5.25))
    plt.title('')
    maxAx = max(max(previsao[:,FORECAST_STATION][:n_size]), max(observado[:,FORECAST_STATION][:n_size]))*1.05
    minAx = min(min(previsao[:,FORECAST_STATION][:n_size]), min(observado[:,FORECAST_STATION][:n_size]))*.95
    plt.scatter(observado[:,FORECAST_STATION][:n_size],
             previsao[:,FORECAST_STATION][:n_size],
             s=80, facecolors='none', edgecolors= (0, 1, 0, .3))
    plt.plot([minAx, maxAx], [minAx, maxAx], 'k-', color = 'r')
    plt.xlim(minAx, maxAx)
    plt.ylim(minAx, maxAx)
    
    plt.xlabel('Observado (m) ')
    plt.ylabel('Previsto (m)')
    plt.savefig(DIR + "0.0 - Previsão vs Observado scatter "+titulo+".png",
            bbox_inches = 'tight', dpi = imageResolution)
    plt.show()
    
    print("NASH "+ np.str(nash(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    print("Acurácia media "+ np.str(accuracyOfMean(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    print("Precisao "+ np.str( precision(previsao[:,FORECAST_STATION]) ))
    print("RMSE "+ np.str(rmse(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    print("MSE "+ np.str(mse(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    print("MAE "+ np.str(mae(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    print("ME "+ np.str(me(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    
    arq = open(DIR + '0.0 - Indicares treinamento ' + titulo + '.csv', 'w')
    arq.write("NASH "+ np.str(nash(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.write("\nAcurácia media "+ np.str(accuracyOfMean(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.write("\nPrecisao "+ np.str( precision(previsao[:,FORECAST_STATION]) ))
    arq.write("\nRMSE "+ np.str(rmse(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.write("\nMSE "+ np.str(mse(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.write("\nMAE "+ np.str(mae(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.write("\nME "+ np.str(me(observado[:,FORECAST_STATION], previsao[:,FORECAST_STATION])))
    arq.close()  

    return observado, previsao

# predição sem plots
def predicao_(scalerx, scalery, model, x_test): 
    #pred = model.predict(scaler.fit_transform(x_test))
    norm_prediction = model.predict(x_test)
    previsao = scalery.inverse_transform(norm_prediction)
    #previsao = model.predict(x_test)

    return previsao, norm_prediction


