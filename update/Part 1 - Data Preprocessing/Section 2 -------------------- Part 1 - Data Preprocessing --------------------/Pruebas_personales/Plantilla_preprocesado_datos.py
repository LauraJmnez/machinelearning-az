# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Plantilla de Pre Procesado

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


#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################

#Al entrenar a la máquina es importante comprobar que funciona con un algoritmo y que no se ha aprendido los datos de memoria (over fitting), por lo que en vez de dos 
#(X e y) variables como hemos tenido hasta ahora, tendremos 4 (x_entrenamiento, x_testing, y_entrenamiento e y_testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados


##############################
#   Escalaado de variable    #
##############################
# Cuando hay una variable con un rango de valores mucho mayor (Salary) y otro con uno menor (Age), hay que normalizarlos para que ambos se muevan en un mismo rango
# y que sea el propio algoritmo el que deba discennir entre qué peso darle a cada variable no por tener un rango mayor o menor, sino porque realmente aportan
# más o menos en el proceso de predicción.
# Hay que distinguir entre estandarización (permite aglutinar valores entorno a la media, tendremos muchos valores cercanos a 0 y pocos alejados de él) 
# y normalización (trasnforma la columna de datos en un conjunto 0-1, el número más pequeño se transforma en 0 y el número mas grande en 1 y el resto se escala de 
# forma lineal)

# Las variables dummy se pueden escalar o no en función de neustro criterio, lo idela es que siempre se estandarice todo, pero esto depende de gustos. Sin embargo,
# la variable dependiente y en este caso no hay que estandarizarla ya que nuestro algoritmo es de clasificación (Compra o no compra). En casos de algoritmos de 
# predicción, como la regresión lineal, sí que se recomineda estandarizarlo también.
# 
#Estandarización:
"""from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
