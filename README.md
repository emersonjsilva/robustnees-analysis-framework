# robustness-analysis-framework
Robustness Analysis Framework

# Primeiros passos
Executar arquivo "Main_Analise_Incerteza.py" no Python
Verificar se as pastas "Anexos_15_min" e "Anexos_120_min" foram criadas

#  Parâmetros do modelo
No arquivo "Main_Analise_Incerteza.py", atentar aos parâmtros


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
