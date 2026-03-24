
# SVR

# Importar dataset
dataset = read.csv('Position_Salaries.csv')
# En R no hace falta distinguir entre variable X e y
'
##########################
# Tratamiento valores NA #
##########################
dataset$Age = ifelse(is.na(dataset$Age), #Si se cumple que haya na
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm =TRUE)), # Hace la media de todos los valores sin tener en cuenta los na
                     dataset$Age) # Si no se cumple, lo deja tal cual
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm =TRUE)),
                        dataset$Salary)

#########################################
# Codificación de variables categóricas #
#########################################

dataset$Country = factor(dataset$Country, 
                         levels = c("France", "Spain", "Germany"), 
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0,1))
'
'
#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) #En este caso hay que poner que porcentaje queremos para el training
# Split lo que hace es poner TRUE en el 80% de los casos
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing
'
'
#########################
#   Esclado de valores
########################
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])
'
# Ajustar SVR con el modelo de datos

library(e1071)
regression = svm(formula = Salary ~ Level, data = dataset,
                 type = "eps-regression",
                 kernel = "radial")


# Predicción
y_pred = predict(regression, newdata = data.frame(Level = 6.5))




# Visualización modelo de regresión SVR
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, newdata = data.frame(Level = dataset$Level))), color = "blue") +
  ggtitle("Predicción (Modelo de regresión)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo ($)")





