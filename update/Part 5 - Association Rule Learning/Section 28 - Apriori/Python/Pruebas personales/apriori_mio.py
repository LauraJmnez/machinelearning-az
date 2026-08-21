# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:53:45 2026

@author: laura
"""

# Algoritmo de recomendacion: Apriori

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocesamiento de los datos
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []  # Lista para almacenar las transacciones
for i in range(0, 7501):  # Iterar sobre cada transacción
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])  # Convertir los datos en transacciones de items

# Entrenamiento del modelo
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)  # Generar las reglas de asociación
'''
min_support: quiero que mire prodcutos que se compren, al menos, 3 veces por día --> 3*7=21 veces a la semana entre el total de las transacciones (21/7500 = 0.003)
'''

# Visualización de resultados
results = list(rules)  # Convertir el resultado en lista
results
results[3]

# Organizar los resultados de manera más legible en un DataFrame de Pandas
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]  # Elemento de la izquierda de la regla
    rhs         = [tuple(result[2][0][1])[0] for result in results]  # Elemento de la derecha de la regla
    supports    = [result[1] for result in results]  # Soporte de la regla
    confidences = [result[2][0][2] for result in results]  # Confianza de la regla
    lifts       = [result[2][0][3] for result in results]  # Lift de la regla
    return list(zip(lhs, rhs, supports, confidences, lifts))  # Combinar todo en una lista

# Convertir los resultados en un DataFrame para facilitar la visualización
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Mostrar los resultados sin ordenar
resultsinDataFrame

# Mostrar los resultados ordenados por el valor de Lift en orden descendente
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')  # Top 10 reglas con mayor Lift
