# Inicializacion de Tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Librerias
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import model_from_json
from datos import data2,clases_1

# Cargar el modelo (.json)
with open("model.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)

# Cargar los pesos (.weights.h5)
model.load_weights("model.weights.h5")
print("Modelo y pesos cargados!")

# Compilar el Modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Cargar datos
datos = data2

# Procesamiento de datos 
max_len = max(len(numero_serial) for numero_serial, clase in datos) # Conversion Datos[i] a ASCII
X_data = np.array([[ord(char) for char in numero_serial.ljust(max_len)] for numero_serial, clase in datos], dtype=np.float32)
X_data = X_data / 255.0  # Normalización 

# Cargar clases
clases = clases_1
y_data = np.array([clases[clase] for numero_serial, clase in datos], dtype=np.int32) # Clases[i] 

# Predicciones
predicciones = []
for numero_serial, tipo_real in datos:
    serial_padded = numero_serial.ljust(max_len)
    serial_ascii = np.array([ord(char) for char in serial_padded], dtype=np.float32) / 255.0
    serial_ascii = np.expand_dims(serial_ascii, axis=0)
    prediccion = model.predict(serial_ascii)
    clase_predicha = np.argmax(prediccion)
    tipo_predicho = list(clases.keys())[clase_predicha]
    predicciones.append((numero_serial, tipo_real, tipo_predicho))

# DataFrame
df_predicciones = pd.DataFrame(predicciones, columns=['Número de Serie', 'Tipo Real', 'Tipo Predicho'])

# Visualización en forma de tabla
print(df_predicciones)
