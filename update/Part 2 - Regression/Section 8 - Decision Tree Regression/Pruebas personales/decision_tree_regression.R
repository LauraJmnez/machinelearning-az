
# Árbol de Decisión para regresión

# Importar dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Mirar NAs
# Tratar variables categóricas

"
# Dividir dataset en conjunto de entrenamiento y conjunto testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing
"
"
# Escalado de valores
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])
"

# Ajustar modelo de regresión
library(rpart)
regression = rpart(formula = Salary ~ ., data = dataset, 
                   control = rpart.control(minsplit = 1))


# Predicción
y_pred = predict(regression, newdata = data.frame(Level = 6.5))



# Visualización modelo de regresión
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = X_grid, y = predict(regression, newdata = data.frame(Level = X_grid))), color = "blue") +
  ggtitle("Predicción con árbol de Decisión (Modelo de regresión)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo ($)")

# No es el mejor modelo


