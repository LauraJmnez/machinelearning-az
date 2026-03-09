# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) #En este caso hay que poner que porcentaje queremos para el training
# Split lo que hace es poner TRUE en el 80% de los casos
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing

#########################
#   Esclado de valores  #
#########################
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])