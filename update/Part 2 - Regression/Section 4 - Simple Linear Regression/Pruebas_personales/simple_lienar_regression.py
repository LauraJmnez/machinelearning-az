# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:49:57 2026

@author: laura
"""

# Regresión Lineal Simple

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import os

#ruta_actual = os.getcwd()
#os.chdir("C:\\Users\\laura\\Desktop\\machinelearning-az\\update\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Pruebas_personales")
# Importar dataset
dataset = pd.read_csv("Salary_Data.csv")

# Crearemos 2 variables:
#   variable X: representarán las variables independeientes del algorítmo (Columna Country, Age y Salary)
#   variable Y: la variable dependiente o la que querámos predecir (columna Purchassed)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################

#Al entrenar a la máquina es importante comprobar que funciona con un algoritmo y que no se ha aprendido los datos de memoria (over fitting), por lo que en vez de dos 
#(X e y) variables como hemos tenido hasta ahora, tendremos 4 (x_entrenamiento, x_testing, y_entrenamiento e y_testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados


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
# En el caso de regresión lineal simple, no es necesario escalado de variables. La librería que usaremos en python hará el trabajo sucio por nosotros. Además, en ese 
# caso solo tenemos una variable independiente (años de experiencia)
#Estandarización:
"""
from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

############################################################################
# Crear modelo de regresión lineal simple con el conjunto de entrenamiento #
############################################################################
from sklearn.linear_model import LinearRegression

regression = LinearRegression() # al ser una Regresion lineal simple no nos hace falta configurar ningún parámetro
regression.fit(X_train, y_train) # tanto X_train como y_test tienen quetener el mismo numerode filas

################################
# Predecir el conjunto de test #
################################
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()