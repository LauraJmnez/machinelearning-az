# Apriori

# Data Preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)


# Se transforma en una matriz dispersa porque la matriz del dataset estará muy vacía
# install.packages('arules') no disponible para la version R4.1.1
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
'
Paso 1:
Para elegir el valor de soporte:
si elegimos como soporte un número muy pequeño, tendrá en ccuentan items que no son muy frecuente.
Debemos buscar un umbral, por ejemplo, teniendo en cuenta el gráfico anterior, quiero tener en cuenta hasta el item papper y mirar dónde llega la barra 
de ese item en el gráfico.
En este caso nos queremos quedar con items que se vendan 3 o 4 veces al día, 3x7=21 ventas a la semana / 7500 cestas de la compra totales =0.0028 (0.003 redondeando)

Para elegir el valor de confianza:
se suele empezar por una confianza por defecto, no demasiado alta, ni muy baja. R nos sugiere empezar por 0.8 (parece alta).

Paso 2 y Paso 3:
Si ejecutamos vemos que nos sale 0 reglas, esto es debido al nivel de confianza tan alto, así que lo bajamos a la mitad y seguimos bajando para ver si el lift aumenta
en el top10

'
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
# Paso 4:
inspect(sort(rules, by = 'lift')[1:10])
