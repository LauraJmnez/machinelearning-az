# Natural Language Pocessing (NLP)

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
'''
Las comillas doble en el fichero pueden dar porblemas, por lo que las podemos ignorar añadiendo el parámetro quoting = 3
'''
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter= "\t", quoting= 3)

## Limpieza de texto
import re
import nltk
nltk.download('stopwords') # Descarga las palabras que son irrelevante
from nltk.corpus import stopwords # Incorporo lalista de palabras descargadas
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # La fundión sub sustituye un parámetro (nos quedamos única y exlusivamente con palabras) por otro, en este caso sustituye por un espacio
    review = review.lower() # Lo deja todo en minúscula
    review = review.split() # Separa las palabras
    ps = PorterStemmer() # Para quedarnos solo con la raíz de ciertas palabras y evitar duplicados.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Me quedo con las palabras de cada review siempe y cuando no esté incluídas en las stopwords y solo la raíz de ellas (love en vez de Loved o loving)
    review = ' '.join(review) # Volvemos a hacer un string con solo con las palabras que nos interesan
    corpus.append(review)

## Crear el Bag of Words
'''
Cogerá cada una de las diferentes palabras y las pondrá en una columna separadas
'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500) # Se podría hacer toda la limpieza de texto directamente con esta función
X = cv.fit_transform(corpus).toarray() # Matriz de características
y = dataset.iloc[:, 1].values

##################################################################################################################
## Agortimo de clasificación Naives Bayes

#dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=0)

#Ajustar el clasificador en el Conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicción de los resultados conel conjunto de testing
y_pred = classifier.predict(X_test)

#matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 73% [(55+91/200) = 0.73]
Precisión de acierto del 56% [55/(55+42) = 0.56]
Recall del 82% [55/(55+12) = 0.82]
F1 Score [2*0.56*0.82/(0.56+0.82) = 0.66]
'''


################################################################################################################
## Algoritmo de clasificación Regresión logística

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 71% [(76+66/200) = 0,71]
Precisión de acierto del 78% [76/(76+21) = 0.78]
Recall del 67% [76/(76+37) = 0.67]
F1 Score [2*0.78*0.67/(0.78+0.67) = 0.72]
'''


################################################################################################################
## Algoritmo de clasificación KNN

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # K suele ser 5
classifier.fit(X_train, y_train)


# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 61% [(74+48/200) = 0,61]
Precisión de acierto del 76% [74/(74+23) = 0.76]
Recall del 57% [74/(74+55) = 0.57]
F1 Score [2*0.76*0.57/(0.76+0.57) = 0.65]
'''


################################################################################################################
## Algoritmo de clasificación SVM

# Ajustar el SVM en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 72% [(74+70/200) = 0,72]
Precisión de acierto del 76% [74/(74+23) = 0.76]
Recall del 69% [74/(74+33) = 0.69]
F1 Score [2*0.76*0.69/(0.76+0.69) = 0.723]
'''


################################################################################################################
## Algoritmo de clasificación Kernel SVM

# Ajustar el clasificador en el Conjunto de Entrenamiento con Kernel Gaussiano RBF
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 73% [(90+57/200) = 0,735]
Precisión de acierto del 92% [90/(90+7) = 0.92]
Recall del 66% [90/(90+46) = 0.66]
F1 Score [2*0.92*0.66/(0.92+0.66) = 0.76]
'''

# Ajustar el clasificador en el Conjunto de Entrenamiento con Kernel sigmoide
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)


# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 72% [(87+58/200) = 0,725]
Precisión de acierto del 89% [87/(87+10) = 0.89]
Recall del 65% [87/(87+45) = 0.65]
F1 Score [2*0.89*0.65/(0.89+0.65) = 0.751]
'''

# Ajustar el clasificador en el Conjunto de Entrenamiento con Kernel polinómico
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)


# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 66% [(43+90/200) = 0,665]
Precisión de acierto del 44% [43/(43+54) = 0.44]
Recall del 76% [43/(43+13) = 0.76]
F1 Score [2*0.44*0.76/(0.44+0.76) = 0.557]
'''
################################################################################################################
## Algoritmo de clasificación Árboles de decisión

# Ajustar el clasificador de árbol de decisión en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state= 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 71% [(74+68/200) = 0,71]
Precisión de acierto del 76% [74/(74+23) = 0.76]
Recall del 67% [74/(74+35) = 0.67]
F1 Score [2*0.76*0.67/(0.76+0.67) = 0.712]
'''

################################################################################################################
## Algoritmo de clasificación Random Forest

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators= 500, criterion= "entropy", random_state=0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 72% [(88+56/200) = 0,72]
Precisión de acierto del 90% [88/(88+9) = 0.90]
Recall del 65% [88/(88+47) = 0.65]
F1 Score [2*0.90*0.65/(0.90+0.65) = 0.754]
'''

#########################################################################################################
## Algoritmo de clasificación CART
#Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', random_state=0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Comparar y_pred con y_test de manera automática con una MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Tenemos una fiabilidad del 65% [(71+59/200) = 0,65]
Precisión de acierto del 73% [71/(71+26) = 0.73]
Recall del 61% [71/(71+44) = 0.61]
F1 Score [2*0.73*0.61/(0.73+0.61) = 0.664]
'''
