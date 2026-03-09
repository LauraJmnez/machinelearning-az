#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Plantilla de Pre Procesado - Datos Categóricos

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

ruta_actual = os.getcwd()
os.chdir("C:\\Users\\laura\\Desktop\\machinelearning-az\\update\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Pruebas_personales")
# Importar dataset
dataset = pd.read_csv("Data.csv")

# Crearemos 2 variables:
#   variable X: representarán las variables independeientes del algorítmo (Columna Country, Age y Salary)
#   variable Y: la variable dependiente o la que querámos predecir (columna Purchassed)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

################################################################################################
#   Codificar datos categóricos --> El objetivo es asignar un numéro a cada valor categórico   #
################################################################################################
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0]) #fit_transfor coge las columnas que yo le indique y las transforma en datos numéricos
X[:, 0] = labelencoder_X.fit_transform(X[:,0]) #Cambiamos los valores categóricos por los numéricos en el dataset

# El problema de esto es que ahora estos valores son comparables y puede inducir error. (O es inferior a 1 y 2 es suerior a 1), para evitar esto existen
# las variables Dummy ("one out encoder" en inglés). Es una forma de traducir una categoría que no tiene un orden a un conjunto de tantas columnas como
# categoría sexisten

# Si las tres variables categóricas son France, Sapin y Germany:
#Country #Age #Salary #Purchased   |         France    Spain    Germany     
#France   44   72000     No        |           1         0         0
#Spain    27   48000     Yes       |           0         1         0
#Germany  30   54000     No        |           0         0         1
#Spain    38   61000     No        |           0         1         0


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct_X = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct_X.fit_transform(X))  # Transformamos X y lo convertimos a un array de NumPy


# Ahora con la Y. En este caso solo hay Yes ('1') y No ('0')
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  # Transformamos las etiquetas categóricas en valores numéricos
