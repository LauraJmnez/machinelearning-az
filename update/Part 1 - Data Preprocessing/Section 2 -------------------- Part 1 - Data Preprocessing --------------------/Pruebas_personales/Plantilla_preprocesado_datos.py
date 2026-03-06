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

#############################################################################
#   Tratamiento NA, vamos a sustituir los NA por la media de cada columna   #
#############################################################################
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy= "mean")
imputer.fit(X[:, 1:3]) # Python no toma el último valor, por eso en vez de 1:2 (solo cogería la columna 1) hay que poner 1:3 para que coja la columna 1 y 2
X[:, 1:3] = imputer.transform(X[:, 1:3])  # Reemplazamos los valores nulos en las columnas seleccionadas
print(X)

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

#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################

#Al entrenar a la máquina es importante comprobar que funciona con un algoritmo y que no se ha aprendido los datos de memoria (over fitting), por lo que en vez de dos 
#(X e y) variables como hemos tenido hasta ahora, tendremos 4 (x_entrenamiento, x_testing, y_entrenamiento e y_testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados


####################################################
#   Escalaado de variable    #
#####################################################
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
from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
