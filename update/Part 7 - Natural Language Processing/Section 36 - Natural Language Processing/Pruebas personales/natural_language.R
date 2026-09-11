
# Natural Language Processing

dataset_original = read.delim( "Restaurant_Reviews.tsv",sep = "\t", quote = '', stringsAsFactors = FALSE)

# Limpieza de textos
#install.packages("tm")
#install.packages("SnowballC")
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # convierte en minúscula todos los parámetros del dataset. tm_map lo cambia en todo el dataset, no solo en una sola fila

#Consultar el primer elemento del corpues
# as.character(corpus[[1]])

corpus = tm_map(corpus, removeNumbers) # Elimina los números
corpus = tm_map(corpus, removePunctuation) # Elimina los signos de puntuación
corpus = tm_map(corpus, removeWords, stopwords(kind = "en")) #eliminar palabras que no nos diga nada
corpus = tm_map(corpus, stemDocument) # Se queda con la raíz de las palabras cuando está de varias formas (Loves,loving,loved --> love)
corpus = tm_map(corpus, stripWhitespace)# Elimina los espacios en blancos adicionales

# Crear la bolsa de plabaras (Bag words)       se creará una columna por cada palabra nueva que aparezca

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999) # Reducir la matrix eliminando aquellas columnas que no aparezcan apenas (Nos quedamos con el 99% más repetidas) Hay que tener cuidado con el porcentaje que se elige


# Algoritmo de clasificación Random Forest
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

dataset$Liked = factor(dataset$Liked, 
                           levels = c(0,1))

## Dividir dataset en conjunto de entrenamiento y conjunto testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80) #En este caso hay que poner que porcentaje queremos para el training
training_set = subset(dataset, split == TRUE) # No quedamos con el 80% (TRUE) para el training
testing_set = subset(dataset, split == FALSE) # Nos quedamos con el 20% (FALSE) para el testing

## Ajustar el clasificador con Random Forest con el conjunto de entrenamiento
library(randomForest)
# classifier = randomForest(formula = Purchased ~ ., data = training_set, ntree = 100)
classifier = randomForest(x = training_set[, -692], 
                          y = training_set$Liked, 
                          ntree = 10)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear matriz de confusión
cm = table(testing_set[, 692], y_pred)
cm
