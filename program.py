# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(os.path.join("preprocessed_CAC40.csv"), delimiter=',', usecols=['Name','Date', 'Open', 'Daily_High', 'Daily_Low',

                                                                                     'Closing_Price'])
da = pd.read_csv(os.path.join("aadr.us.txt"), delimiter=',', usecols=['Date', 'Open', 'High', 'Low',

                                                                                     'Close'])
# Data saved to : stock_market_data-AAL.csv

# on trie le dataframe par date
df = df.sort_values('Date')


df.head() #par défaut, affiche les 5 premières colonnes du datframe

plt.figure(figsize=(18, 9))
plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

# Calcul du prix moyen du cours sur une journée

highPrices = df.loc[:,'High'].to_numpy()
lowPrices = df.loc[:,'Low'].to_numpy()
midPrices = (highPrices+lowPrices)/2.0

# On sépare les données de formation que l'on va apprendre au programme
# et les données de test. On sépare aux deux moitiés

trainData = midPrices[:1100]
testData = midPrices[1100:]

#je dois maintenant définir un scaler pour normaliser les 
#données. MinMaxScalar met à l'échelle toutes les données 
#pour qu'elles soient comprises entre 0 et 1

scaler = MinMaxScaler()
trainData = trainData.reshape(-1,1) #on reshape pour qu'il sorte la dim qu'on veut => applatit le tableau
testData = testData.reshape(-1,1)


# Entraîner le Scaler avec des données d'entraînement et on lisse les données
smoothing_window_size = 25 #on choisit une taille de 250 pour ne pas
#dépasser la taille des bits en jeu

for di in range(0,1000,smoothing_window_size):
    scaler.fit(trainData[di:di+smoothing_window_size,:]) # on fait que sur les données d'entrainement
    trainData[di:di+smoothing_window_size,:] = scaler.transform(trainData[di:di+smoothing_window_size,:])

# On normalise le dernier bit des données restantes
scaler.fit(trainData[di+smoothing_window_size:,:])
trainData[di+smoothing_window_size:,:] = scaler.transform(trainData[di+smoothing_window_size:,:])

# On regroupe les données d'entrainements et de test
trainData = trainData.reshape(-1)

# On normalise les données de test
testData = scaler.transform(testData).reshape(-1)

# on Effectue maintenant le lissage de la moyenne mobile exponentielle.
# Ainsi, les données auront une courbe plus lisse que les données originales en dents de scie.
MME = 0.0 #on définit notre variable de moyenne mobile exponentielle
gamma = 0.1 #gamma décide de la contribution de la 
#prédiction la plus récente à la MME. Ici les données les plus récentes ont 
#un poids de 10%

for ti in range(1100):
  MME = gamma*trainData[ti] + (1-gamma)*MME
  trainData[ti] = MME

# On concatène dans un même tableau les données d'entrainements et de test pour les comparer
all_mid_data = np.concatenate([trainData,testData],axis=0)

window_size = 100
N = trainData.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []


# on teste l'efficacité de la moyenne mobile simple
for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime('%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(trainData[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-trainData[pred_idx])**2)
    std_avg_x.append(date)
    
print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))#erreur fournie par sklearn

#Tracé de la superposition du cours réel avec la prédiction sur les 1500
#premières dates

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='Cours reel')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Prix moyen')
plt.legend(fontsize=18)
plt.show()


#################################################################################
#################################################################################
###################### Méthode moyenne mobile exponentielle #####################
#################################################################################
#################################################################################

window_size = 1000
N = trainData.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0 #moyenne courante

run_avg_predictions.append(running_mean)

decroissance = 0.5 #décroissance

#on teste maintenant le calcul de moyenne mobile exponentielle

for pred_idx in range(1,N):

    running_mean = running_mean*decroissance + (1.0-decroissance)*trainData[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-trainData[pred_idx])**2)
    run_avg_x.append(date)

print('Erreur quadratique moyenne pour ce type de calcul de moyenne: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()