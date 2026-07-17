# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:35:36 2026

@author: laura
"""

# Clustering Jerárquico
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importar dataset
dataset = pd.read_csv("Mall_Customers.csv")

# Crearemos 1 variable con las columnas que nos interesa:
X = dataset.iloc[:, [3,4]].values

# Utilizar el dendograma par encontrar el número óptimo de clústers
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method= "ward"))
plt.title("Dendograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Al trazar una línea imaginaria en la rama más larga que no esté cortado horizontalmente por otra (alrededor de la distancia euclidea 235 aprox) obtenemos 5 clústeres


# Ajustar el clustering jerárquico a nuestro conjunto de datos
# clustering aglomerativo
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)


# Visualización de los clústers 2D
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.title("Clúster de clientes")
plt.xlabel("Ingresos anuales (miles de $)")
plt.ylabel("Puntuación de gastos")
plt.legend()
plt.show()
