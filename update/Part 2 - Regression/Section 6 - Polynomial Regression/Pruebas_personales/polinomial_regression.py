# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:39:38 2026

@author: laura
"""

#Regresión polinómica

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Crearemos 2 variables:
#   variable X: representarán las variables independeientes del algorítmo (Columna Level)
#   variable Y: la variable dependiente o la que querámos predecir (columna Salary)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################
#En este caso, como tenemos un conjunto de datos muy pequeño, no vamos a dividir entre entrenamiento y testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados

"""
"""
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
# Cuando hablamos de regresion lineal simple, no hace falta ningún tipo de escalado

from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

## Ajustar la regresión Lineal con el dataset
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualización
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()
# Como vemos este modelo no es el más preciso para predecir, el CEO estaría muy infravalorado (por ejemplo)

# Predicción de nuestros modelos
lin_reg.predict([[6.5]])
# Out[47]: array([330378.78787879])     -> Definitivamente no es el mejor modelo, Segun el modelo con un nivel 6.5 cobraría muchisimo mas de lo que le corresponde realmente









## Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2) # Creamos tantos cuadrados como grados de regresión querramos, en este caso: regresion hasta garado 2
X_poly = poly_reg.fit_transform(X) # Se añade al principio la constante (col 1), después nuestra variable en sí (col 2), y luego el cuadrado de nuestra variable (col 3)

lin_reg_2 = LinearRegression() # El truco de la reg. polinómica es alterar el conjunto de datos con los grados de regresión que quiero.
lin_reg_2.fit(X_poly, y)

# Visualización
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
plt.title("Modelo de regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()

# Si en vez de grado 2, lo hubieramos hecho con un grado 3, hubiera mejorado?
poly_reg = PolynomialFeatures(degree = 3) # Creamos tantos cuadrados como grados de regresión querramos, en este caso: regresion hasta garado 2
X_poly = poly_reg.fit_transform(X) # Se añade al principio la constante (col 1), después nuestra variable en sí (col 2), y luego el cuadrado de nuestra variable (col 3)

lin_reg_2 = LinearRegression() # El truco de la reg. polinómica es alterar el conjunto de datos con los grados de regresión que quiero.
lin_reg_2.fit(X_poly, y)

# Visualización
#X_grid =np.arange(min(X), max(X), 0.1) # para evitar que se vean rectas entre punto, dandole valores intermedios (opcional)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo ($)")
plt.show()
# Mejoraría muchísimo

# Predicción de nuestros modelos
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
# Out[50]: array([174878.07765118])   --> Mucho más hacertado
















