# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:33:10 2026

@author: laura
"""

# Regresión con Árboles de Decisión

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#############################
#  Preprocesado de dataset  #
#############################

# Tratamiento de NA

# Tratamiento de Variables categóricas
"""
# Dividir dataset en conjunto de entrenamiento y conjunto testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados
"""
# Escalaado de variable
"""
from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#######################
# Modelo de Regresión #
#######################

# Ajustar el modelo de regresión con el dataser (Lineal o polinómica)
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)
# Predicción de nuestros modelos
y_pred = regression.predict([[6.5]])


# Visualización de resultados del Modelo polinómico
# X_grid =np.arange(min(X), max(X), 0.1) # Suaviza las curvas
# X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición empleado")
plt.ylabel("Sueldo ($)")
plt.show()
