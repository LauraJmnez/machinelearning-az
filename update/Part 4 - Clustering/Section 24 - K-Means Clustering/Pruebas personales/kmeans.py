# -*- coding: utf-8 -*-
"""
Created on Thu May  7 07:53:56 2026

@author: laura
"""

# K-Means

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,[3,4]].values

#Método del Codo para averiguar el número óptimo de clústeres
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, 
                    n_init = 10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de clústers")
plt.ylabel("WCSS(k)")
plt.show
    #Sale que el númeor óptimo es k=5

# Aplicar el método de k-means con k=5 clústers para segmentar el dataset
kmeans = KMeans(n_clusters=5, init = "k-means++", max_iter = 300, 
                    n_init = 10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Representación gráfica de los clústers
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 200, c = "black", label = "Baricentros")
plt.title("Clúster de clientes")
plt.xlabel("Ingresos anuales (miles de $)")
plt.ylabel("Puntuación de gastos")
plt.legend()
plt.show