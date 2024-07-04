# Inicializacion de Tensorflow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from datos import data1,clases_1 # Importar datos

# Cargar datos
datos = data1

# Procesamiento de datos 
max_len = max(len(numero_serial) for numero_serial, clase in datos) # Datos[i] 
X_data = np.array([[ord(char) for char in numero_serial.ljust(max_len)] for numero_serial, clase in datos], dtype=np.float32) # Conversion Datos[i] a ASCII

# Cargar Clases
clases = clases_1

y_data = np.array([clases[clase] for numero_serial, clase in datos], dtype=np.int32) # Clases[i] 

# Normalización 
X_data = X_data / 255.0  # Normalizar al rango [0, 1]

# Inicio RNA 
# Arquitectura RNA 
model = Sequential()
model.add(Dense(60, input_dim=max_len, activation='relu')) # Capa Oculta 1, activación ReLU
model.add(Dense(40, activation='relu'))                    # Capa Oculta 2, activación ReLU
model.add(Dense(32, activation='relu'))                    # Capa Oculta 3, activación ReLU
model.add(Dense(15, activation='softmax'))                 # Capa de Salida, activación Softmax para clasificación multiclase

# Argumentos para el aprendizaje RNA
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenamiento RNA
history = model.fit(X_data, y_data, epochs=20, batch_size=1, verbose=1)

# Gráficos de precisión y pérdida
plt.figure(figsize=(8, 4))

# Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
# Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()

# Prediccion 
predicciones = []
for numero_serial, tipo_real in datos:
    serial_padded = numero_serial.ljust(max_len)
    serial_ascii = np.array([ord(char) for char in serial_padded], dtype=np.float32) / 255.0
    serial_ascii = np.expand_dims(serial_ascii, axis=0)  # Añadir dimensión para predicción
    prediccion = model.predict(serial_ascii)
    clase_predicha = np.argmax(prediccion)
    tipo_predicho = list(clases.keys())[clase_predicha]
    predicciones.append((numero_serial, tipo_real, tipo_predicho))

# Data Frame
df_predicciones = pd.DataFrame(predicciones, columns=['Número de Serie', 'Tipo Real', 'Tipo Predicho'])

# Visualización en forma de tabla
print(df_predicciones)

# Visualización en forma de tabla
#plt.figure(figsize=(8, 4))
#plt.table(cellText=df_predicciones.values, 
#          colLabels=df_predicciones.columns, 
#          cellLoc='center', loc='upper center')
#plt.title('Predicciones del Modelo para Números de Serie')
#plt.axis('off')
plt.show()

# Serializar el modelo a JSON
model_json = model.to_json()
with open("model_020000.json", "w") as json_file:
    json_file.write(model_json)

# Serializar los pesos a HDF5
model.save_weights("model_020000.weights.h5")
print("Modelo Guardado!")
