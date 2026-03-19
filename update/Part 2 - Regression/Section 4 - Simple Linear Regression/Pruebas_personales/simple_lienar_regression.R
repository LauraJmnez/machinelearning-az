# Importar dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,2:3]
# En R no hace falta distinguir entre variable X e y

#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) #En este caso hay que poner que porcentaje queremos para el training
# Split lo que hace es poner TRUE en el 80% de los casos
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing

#########################
#   Esclado de valores
########################
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

#############################################################
# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
summary(regressor)
#En este caso, cuanto más asteristicos hay, mas significativo es el modelo. Por o general se recomienda que sea inf al5%

#############################################
#Predecir resultados con el conjunto test

y_pred = predict(regressor, newdata = testing_set) #las columnas se deben llamar exactamente igual a las que usamos para crear la predicción

#Visualización conjunto de entrenamiento
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = "red") +
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs. Años de Experiencia (Conjunto de Entrenamiento)") +
  xlab("Años de Experiencia") +
  ylab( "Sueldo ($)")

#Visualización conjunto de testing  
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             color = "red") +
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regressor, newdata = training_set)),
            colour = "blue") +
  ggtitle("Sueldo vs. Años de Experiencia (Conjunto de Testing)") +
  xlab("Años de Experiencia") +
  ylab( "Sueldo ($)")


