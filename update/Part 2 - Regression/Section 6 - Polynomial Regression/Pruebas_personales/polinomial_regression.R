## Regresión polinómica
"""
El objetivo es ver si el nuevo empleado que quieren contratar ha dicho la verdad con respecto a su sueldo actual
para ver qué sueldo se le podría ofrecer en esta nueva empresa
"""
# Importar dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# No hay NAs
# No hay variables categóricas
# Tampoco vamos a escalar, no es necesario

library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = "red") +
  ggtitle("Sueldo vs. Puesto laboral") +
  xlab("Nivel laboral") +
  ylab( "Sueldo ($)")
# Con esta visualización simple, vemos que no es lineal

# Como nuestro dataset es muy pequeño, no vamos a dividir en entrenamiento y test

###########################################
# Ajustar modelo de regresión lineal #
##########################################
lin_reg = lm(formula = Salary ~ ., data = dataset)


library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = "blue") +
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo ($)")
  
# Predicción
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))


###########################################
# Ajustar modelo de regresión polinómica #
##########################################
dataset2 = dataset
dataset2$Level2 = dataset$Level^2 #Crear las variables al cuadrado, cubo o las que sea
# dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~ ., data = dataset2)
summary(poly_reg)


ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset2)), color = "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo ($)")

#Intentamos mejorarlos añadiendo más cuadrados
dataset3 = dataset
dataset3$Level2 = dataset$Level^2
dataset3$Level3 = dataset$Level^3
dataset3$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset3)
summary(poly_reg)

X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = X_grid, y = predict(poly_reg, newdata = data.frame(Level = X_grid,
                                                                       Level2 = X_grid^2,
                                                                       Level3 = X_grid^3,
                                                                       Level4 = X_grid^4))), color = "blue") +
  ggtitle("Predicción polinómica del sueldo en función del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo ($)")

# Predicción
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))
# EL modelo polinomico se ajusta bastante bien para evitar mentiras en las entrevistas
