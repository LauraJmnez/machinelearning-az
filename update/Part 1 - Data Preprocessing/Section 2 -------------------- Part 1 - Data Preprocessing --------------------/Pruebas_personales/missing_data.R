# Data Preprocessing Template - Datos faltantes

# Importing the dataset
dataset = read.csv('Data.csv')

##########################
# Tratamiento valores NA #
##########################
dataset$Age = ifelse(is.na(dataset$Age), #Si se cumple que haya na
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm =TRUE)), # Hace la media de todos los valores sin tener en cuenta los na
                     dataset$Age) # Si no se cumple, lo deja tal cual
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm =TRUE)),
                        dataset$Salary)
