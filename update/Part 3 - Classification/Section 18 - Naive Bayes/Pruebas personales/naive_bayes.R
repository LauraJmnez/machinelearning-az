# Naive Bayes

dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

'''
#Codificar la variable de clasificación como factor
dataset$Purchased = factor(dataset$Purchased, 
                           levels = c(0,1))
'''

## Dividir dataset en conjunto de entrenamiento y conjunto testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75) #En este caso hay que poner que porcentaje queremos para el training
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing


## Escalado de valores
training_set[,1:2] = scale(training_set[,1:2])
testing_set[,1:2] = scale(testing_set[,1:2])


## Ajustar el clasificador con el conjunto de entrenamiento
library(e1071)
classifier = naiveBayes(formula = Purchased ~ ., data = training_set)
classifier = naiveBayes(x = training_set[,-3], y = training_set$Purchased)



# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-3])

# Crear matriz de confusión
cm = table(testing_set[, 3], y_pred) # La predicción falla en un 7+7=10 (14%)


# Visualización del Conjunto de Entrenamiento
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Naïve Bayes (Conjunto de Training)',
     xlab = 'Edad', ylab = 'Sueldo Estimado', 
     xlim =range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), 
        add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualización del Conjunto de Testing
library(ElemStatLearn)
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Naïve Bayes (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado', 
     xlim =range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), 
        add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

