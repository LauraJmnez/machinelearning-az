# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:35:21 2026

@author: laura
"""

# SVM

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

ruta_actual = os.getcwd()

# Importar dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# Crearemos 2 variables:
#   variable X: representarán las variables independientes del algorítmo (Columna Country, Age y Salary)
#   variable Y: la variable dependiente o la que querámos predecir (columna Purchassed)
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values



# Dividir dataset en conjunto de entrenamiento y conjunto testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados


# Escalado de variable
from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Ajustar el SVM en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, y_train)



# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
array([[66,  2],
       [ 8, 24]], dtype=int64) Solo nos equivicamos en un 8+2 = 10 (10%)

'''
# Representación gráfica de los resultados del algoritmo en el Conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0]. max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1]. max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show

# Representación gráfica de los resultados del algoritmo en el Conjunto de testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0]. max() + 1, step = 0.01), #con meshgrid se pintan muchos puntitos (pixeles) muy pequeños a muy poca distancia
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1]. max() + 1, step = 0.01)) # haciendo que se coloree el fondo
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Testing set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show








