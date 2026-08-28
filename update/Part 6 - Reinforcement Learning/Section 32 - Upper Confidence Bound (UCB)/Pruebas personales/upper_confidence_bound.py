# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:29:06 2026

@author: laura
"""

# Upper confident Bound (UCB)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Cargar el dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

## Algoritmo de UCB

# Paso 1:
N = 10000  # Número total de rondas
d = 10  # Número de anuncios
ads_selected = []  # Lista para almacenar los anuncios seleccionados
numbers_of_selections = [0] * d  # Contador de cuántas veces se ha seleccionado cada anuncio
sums_of_rewards = [0] * d  # Suma de las recompensas obtenidas por cada anuncio
total_reward = 0  # Recompensa total acumulada
# Paso 2:
import math
for n in range(0, N):  # Para cada ronda
    ad = 0  # Inicializar el anuncio seleccionado
    max_upper_bound = 0  # Inicializar el límite superior máximo
    for i in range(0, d):  # Evaluar cada anuncio
        if (numbers_of_selections[i] > 0):  # Si el anuncio ha sido seleccionado al menos una vez
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]  # Calcular la recompensa media del anuncio
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])  # Cálculo del delta
            upper_bound = average_reward + delta_i  # Calcular el límite superior
        else:
            upper_bound = 1e400  # Asignar un valor muy grande si el anuncio no ha sido seleccionado aún
        # Seleccionar el anuncio con el mayor límite superior
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)  # Añadir el anuncio seleccionado a la lista
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1  # Incrementar el contador de selecciones del anuncio
    reward = dataset.values[n, ad]  # Obtener la recompensa (clic) del anuncio seleccionado
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  # Sumar la recompensa al total del anuncio
    total_reward = total_reward + reward  # Sumar la recompensa total

# Visualización de los resultados
plt.hist(ads_selected)  # Crear el histograma de las selecciones de anuncios
plt.title('Histograma de selecciones de anuncios')  # Título del gráfico
plt.xlabel('Anuncios')  # Etiqueta en el eje X
plt.ylabel('Número de veces que se seleccionó cada anuncio')  # Etiqueta en el eje Y
plt.show()  # Mostrar el gráfico
    
# Nos sale que el mejor es el anuncio 4+1. Es decir, el anuncio 5