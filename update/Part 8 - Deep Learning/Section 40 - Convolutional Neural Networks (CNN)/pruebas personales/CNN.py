# Redes Neuronales Convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forges keras (desde consola anaconda.) 

'''
En este caso no hay que hacer preprocesado de datos puesto que lo que vamos a tener es un directorio llamado dataset con 10.000 imágenes divididas en 4.000 imágenes de entrenamiento y 1.000 imágenes de testing para
la categoría perro y para la categoría gato.
Los datos inroducidos se pueden organizar, etiquetar y utomatizar con la librería keras. 
'''

# Parte 1 - Construir el modelo de CNN

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Iniciar la RNA
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                      input_shape = (64,64,3), activation = "relu"))
'''
Empezar con 32 detectores de características de tamaño 3x3 . Es mejor empezar desde poco (Las potencias de dos suelen ser las mejores) y ya luego iremos mejorando nuestros resultados con más filtros
Intentar mantener las imágenes con un tamaño pequeño para evitar el coste computacional.
En el caso de los perros, no todos tienen los mismos colores, por lo que tenemos que ver si nos interesa o no añadir canales de color # input_shape(n.filas,n.colunas,n.canales de color)
'''

# Paso 2 - Max Pooling

classifier.add(MaxPooling2D(pool_ize = (2,2)))

# Añadimos otra capa de convolución y max pooling (filtrando 2 veces) para mejorar los resultados:
'''
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_ize = (2,2)))
'''

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Conection
classifier.add(Dense(units = 128, activation = "relu")) #capa oculta
# classifier.add(Dense(units = 128, activation = "relu")) #capa oculta
classifier.add(Dense(units = 1, activation = "sigmoid")) #capa salida
'''
como units tenemos que escoger un número que no sea ni muy pequeño ni muy grande. 128 parece un número corecto. Luego sino podemos añadir más.
Podemos añadir tantas capas como queramos, pero hay que recordar que hay que terminar en una capa de salida, y esta quiere probabilidades, por lo que la función de activación sería 'sigmoid'. Como units ponemos
1 en vez de 2 (perro y gato) porque en nuestro caso es un problema de clasificación binario
'''

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# Parte 2 - Ajustar la CNN a las imágenes para entrenar
from keras.preprocesing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(recale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_generator = train_datagen.flow_from_directory('dataset/train_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

testing_generator = test_datagen.flow_from_directory('dataset/test_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

classifier.fit_generator(training_generator, steps_per_epoch = 8000, epochs = 30, validation_data = testing_generator, validation_steps = 2000)
