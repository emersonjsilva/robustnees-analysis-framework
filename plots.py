# -*- coding: utf-8 -*-
"""

@author: Emerson Jean da Silva
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime


resol_img = 200

def histograma(col, nome_arquivo):
    data_e_hora_atuais = datetime.now()
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
    
    
    nome = nome_arquivo
    nome_saida = "_col_"+np.str(col)	
    
    #x = np.loadtxt(nome,unpack=True,usecols=[col], delimiter=",")#textfile
    x = pd.read_csv(nome+'.csv',   delimiter=",")
    #print(x.head())
    x = np.array(x)
    x = x[:, col]
    
    media = np.average(x)
    mediana = np.median(x)
    
    
    plt.close('all')
    plt.figure(figsize=(14, 8))
    n, y, _t = plt.hist(x, bins='auto', histtype='stepfilled', facecolor='lime')
    #print(np.where(n == np.amax(n)))
    j = np.where(n == np.amax(n))
    #print(j)
    i = np.int(np.double(j[0][0]))
    
    md = (y[i]+y[i+1])/2
    plt.axvline(media, color='blue', linewidth=3,label='Média')
    plt.axvline(md, color='red', linewidth=3, label='Moda')
    plt.axvline(mediana, color='green', linewidth=3, label='Mediana')
    plt.legend(loc='upper right')
    plt.savefig(nome + nome_saida + "_hist.png",format='png', dpi=1600)
    #plt.show()
    
    print("Gerado histrograma: "+nome + nome_saida + "_hist.png")
    	
    #moda, freq = st.mode(x)
    #moda = np.double(moda)
    arq = open(nome+'_stats.txt', 'a')
    texto =         "\nNome arquivo; " + nome + nome_saida + "_hist.png"\
                    +"\nData hora; " + data_e_hora_em_texto\
                    +"\nNumero de amostras; " + np.str(np.size(x))\
                    +"\nMedia; " + np.str(media)\
                    +"\nDesvio padrão; " + np.str(np.std(x))\
                    +"\nMediana; " + np.str(mediana)\
                    +"\nClasse Modal; " + np.str(md)\
                    +"\nMaximo: " + np.str(np.max(x))\
                    +"\nMinimo: " + np.str(np.min(x))\
                    +"\n\n-------------------------------------------------------\n"            
    arq.write(texto)
    arq.close()
    print("finalizado arquivo estatístico: stats.txt")
    
                    #+"\nCurtose; " + np.str(st.kurtosis(x))\
                    #+"\nAssimetria; " + np.str(st.skew(x))\
    return x


def plot_previsao(nome):
	 
    df = pd.read_csv(nome+".csv",   delimiter=",")
    df = np.array(df)
    
    '''
    fn = nome_arquivo+'.csv'
    x1 = np.loadtxt("previsao.csv",unpack=True,usecols=[0], delimiter=",")#textfile
    x2 = np.loadtxt("previsao.csv",unpack=True,usecols=[1], delimiter=",")#textfile
    '''
    i1 = np.arange(0, np.size(df[1,:]),1)
    
    print(np.size(df[1,:]))
    #Plota erro da rede em função das epocas de treinamento
    plt.figure(1)
    '''
    plt.plot(i1, x[0, :], label="Target", color = 'red')
    plt.plot(i1, x[1, :], label="Predicition", color = 'green')
    plt.legend()
    '''
    
    x = df[1:,0]
    y = df[1:,1]
    
    xmin = 0
    xmax = 500
    ymin = 0
    ymax = 500
    
    plt.close('all')
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    media = np.average(x)
    mediana = np.median(x)
    n, y_, _t = plt.hist(x, bins='auto', histtype='stepfilled', facecolor='lime')
    plt.axis([xmin, xmax, ymin, ymax])
    #print(np.where(n == np.amax(n)))
    j = np.where(n == np.amax(n))
    #print(j)
    i = np.int(np.double(j[0][0]))
    
    md = (y_[i]+y_[i+1])/2
    plt.axvline(media, color='blue', linewidth=3,label='Média')
    plt.axvline(md, color='red', linewidth=3, label='Moda')
    plt.axvline(mediana, color='green', linewidth=3, label='Mediana')
    plt.legend(loc='upper right')
    
    plt.subplot(122)
    media = np.average(y)
    mediana = np.median(y)
    n, y_, _t = plt.hist(y, bins='auto', histtype='stepfilled', facecolor='lime')
    plt.axis([xmin, xmax, ymin, ymax])
    #print(np.where(n == np.amax(n)))
    j = np.where(n == np.amax(n))
    #print(j)
    i = np.int(np.double(j[0][0]))
    
    md = (y_[i]+y_[i+1])/2
    plt.axvline(media, color='blue', linewidth=3,label='Média')
    plt.axvline(md, color='red', linewidth=3, label='Moda')
    plt.axvline(mediana, color='green', linewidth=3, label='Mediana')
    plt.legend(loc='upper right')    
    plt.savefig(nome + "_hist.png", dpi = 1200, bbox_inches = 'tight', pad_inches = 0.1)	
    
    
def plotTimeSeries(dados, titulo, xtitulo, ytitulo, nomeArquivo):
        dados.plot(figsize=(10.5, 5.25))       
        plt.title(titulo) 
        plt.ylabel(ytitulo)
        plt.xlabel(xtitulo)
        plt.legend()
        plt.savefig(nomeArquivo, bbox_inches = 'tight', dpi = resol_img)
        plt.show()
                
   
def plotStations(dataFrameDadosEstacoes, nomesEstacoes, unidadeMedidaEstacoes,
                 nomeArquivo):
    estacao = 0
    while(estacao < len(nomesEstacoes)):
        arrayDadosEntrada = np.array(dataFrameDadosEstacoes)
        
        plt.close('all')   
        #plt.figure(figsize=(10.5, 5.25))
        plt.title('Histograma da estação \"'+ np.str(nomesEstacoes[estacao])+'\"')
        n, bins1, patches1 = plt.hist(arrayDadosEntrada[:, estacao], bins=100, histtype='stepfilled',
                             facecolor='lime')
        if(unidadeMedidaEstacoes[estacao] == 'Precipitação [mm]'):
            plt.yscale('log')
        plt.xlabel(unidadeMedidaEstacoes[estacao])
        plt.ylabel('Frequência')
        
        plt.savefig(nomeArquivo
                    + np.str(nomesEstacoes[estacao])+".png",
                    bbox_inches = 'tight', dpi = resol_img)
        plt.show()
        estacao = estacao + 1

# Fim 0.1 - Plota o histograma das estações

'''
def plotScatter(mainTitle, x, y, xtitle, ytitle, nameFile, scale_):
    plt.close('all')
    plt.figure(figsize=(12.5, 5.25))
    plt.title(mainTitle)
    plt.plot(x, y, 'o', color=(.5, .5, .9, .1))
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis(scale_)
    plt.savefig(nameFile + ".png",
            bbox_inches = 'tight', dpi = resol_img)
    plt.show()
'''