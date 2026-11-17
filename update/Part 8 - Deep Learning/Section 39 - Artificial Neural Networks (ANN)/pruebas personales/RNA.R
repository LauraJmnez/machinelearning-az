# Redes Neuronales Artificiales

# Importar dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]

# Codificar los factores para a RNA
dataset$Geography = as.numeric(factor(dataset$Geography, 
                                      levels = c("France", "Spain", "Germany"), 
                                      labels = c(1,2,3)))

dataset$Gender = as.numeric(factor(dataset$Gender, 
                                   levels = c("Female", "Male"), 
                                   labels = c(0,1)))

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio =0.8)
training_set = subset(dataset, split = TRUE)
testing_set = subset(dataset, split = TRUE)


# Escalado de variables
training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])

# Crear la Red Neuronal
# install.packages("h2o")
library(h2o)
h2o.init(nthreads = -2) # nos podríamos conectar con algún servidor definiendo el parámetro ip y/o un port
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = training_set, # si no funcion usar as.h2o(trainig_set)
                              activation= "Rectifier",
                              hidden = c(6,6), # Como truco se recomienda usar el promedio entre la capa de entrada y la capa de salida (10 + 1)/2 = 5.5 aprox 6). En ste caso estamos añadiendo 2 capas ocultas, ambas con 6 nodos
                              epochs = 100,
                              train_samples_per_iteration = -2)
                              
                              
# Predicción de los resultados con el conjunto de testing
prob_pred = h2o.predict(classifier, newdata = as.h2o(testing_set[,-11]))
y_pred = ifelse(prob_pred> 0.5)
y_pred = as.vector(y_pred)

# Crear la matriz de confusión
cm = table(testing_set[, 11], y_pred)

'''
Resultado:
1535+190/2000 =0.8625 (86%)
'''
# Cerrar la sesión de H2O
h2o.shutdown()