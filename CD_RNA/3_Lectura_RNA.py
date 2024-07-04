import cv2
from pyzbar import pyzbar
import numpy as np
from keras.models import model_from_json
from datos import data2,clases_2


# Cargar el modelo (.json)
with open("model.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)
# Cargar los pesos (.weights.h5)
model.load_weights("model.weights.h5")
print("Modelo y pesos cargados!")

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Cargar clases (Para Visiluzacion)
clases = clases_2

# Función 
def leer_codigos_de_barras():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar códigos de barras
        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            # Obtencion datos del código de barras
            barcodeData = barcode.data.decode("utf-8")

            # Prepararcion datos
            max_len = len(barcodeData)  # Longitud máxima del número de serie (ajustado según necesidad)
            serial_padded = barcodeData.ljust(max_len)
            serial_ascii = np.array([ord(char) for char in serial_padded], dtype=np.float32) / 255.0
            serial_ascii = np.expand_dims(serial_ascii, axis=0)  # Añadir dimensión para predicción

            # Hacer la predicción con el modelo si hay datos válidos
            if len(serial_ascii) > 0:
                prediccion = model.predict(serial_ascii)
                clase_predicha = np.argmax(prediccion)
                producto_predicho = clases.get(clase_predicha, "Desconocido") 

                # Mostrar rectángulo código de barras
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Mostrar número de código de barras
                cv2.putText(frame, barcodeData, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Mostrar el producto predicho 
                cv2.putText(frame, "Producto: {}".format(producto_predicho), (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Mostrar el frame
        cv2.imshow('Lectura de Código de Barras', frame)

        # Salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función
leer_codigos_de_barras()
