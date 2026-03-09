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

#############################################################################
#   Tratamiento NA, vamos a sustituir los NA por la media de cada columna   #
#############################################################################
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= "mean")
imputer.fit(X[:, 1:3]) # Python no toma el último valor, por eso en vez de 1:2 (solo cogería la columna 1) hay que poner 1:3 para que coja la columna 1 y 2
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Reemplazamos los valores nulos en las columnas seleccionadas
print(X)
