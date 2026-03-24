# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:32:51 2026

@author: laura
"""

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))



#######################
# Modelo de Regresión #
#######################

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)


# Predicción de nuestros modelos con SVR
"""
Como había mucha direncia entre el C-level y el CEO (CEO solo hay uno tambien), el modelo al escalarlo lo reconoce como 
outlayer y no lo tiene en cuenta, por eso se obtiene u modelo bien ajustado para todos menos para el CEO 
"""
y_pred = regression.predict(sc_X.transform([[6.5]]))

# Como los datos están escalados, los resultados se han visto afectados también. Hay que revertirlo:
y_pred = sc_y.inverse_transform(y_pred)
    

# Visualización de resultados del SVR con el escalado invertido
X_grid =np.arange(min(X), max(X), 0.1) # Suaviza las curvas
X_grid = X_grid.reshape(len(X_grid), 1)

#Sin X_grid
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regression.predict(X).reshape(-1, 1)), color = "blue")
plt.ticklabel_format(style='plain', axis='y') # Quita la notación científica
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición empleado")
plt.ylabel("Sueldo ($)")
plt.show()

#Con X_grid
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regression.predict(X_grid).reshape(-1, 1)), color = "blue")
plt.ticklabel_format(style='plain', axis='y') # Quita la notación científica
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición empleado")
plt.ylabel("Sueldo ($)")
plt.show()