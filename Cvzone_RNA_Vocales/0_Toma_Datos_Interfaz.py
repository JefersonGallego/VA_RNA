import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from cvzone.Utils import rotateImage


# Crear el directorio 'Data_Vocal' si no existe
if not os.path.exists('Data_0'):
    os.makedirs('Data_0')

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Opcional: establecer la resolución de captura de la cámara (si la cámara lo soporta)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 0 = 1280x720 1 y 2 = 640x480  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# Crear una ventana para la cámara y otra para mostrar el cuadro delimitador
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)  # Nueva ventana para mostrar la ROI

# Establecer el tamaño de la ventana de la imagen
cv2.resizeWindow("Image", 1280, 720)  # Ajustar según sea necesario

# Función para dibujar el cuadro delimitador
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    cv2.putText(img, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

bbox = None
drawing = False
data_counter = 0  # Contador para las imágenes y datos guardados
frame = None  # Definir la variable frame fuera de cualquier función

# Función para manejar eventos del mouse
def mouse_callback(event, x, y, flags, param):
    global bbox, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [x, y, 0, 0]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            bbox[2] = x - bbox[0]
            bbox[3] = y - bbox[1]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if bbox[2] < 0:
            bbox[0] += bbox[2]
            bbox[2] *= -1
        if bbox[3] < 0:
            bbox[1] += bbox[3]
            bbox[3] *= -1

# Asignar la función de callback del mouse a la ventana de la cámara
cv2.setMouseCallback("Camera", mouse_callback)

# Función para capturar y guardar la imagen
def capture_image():
    global data_counter, bbox, frame
    if bbox is not None and frame is not None:
        x, y, w, h = bbox
        roi = frame[int(y):int(y+h), int(x):int(x+w)]

        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (30, 30))
            roi_vector = roi_resized.flatten()

            img_filename = f"Data_0/image_{data_counter}.png"
            cv2.imwrite(img_filename, roi_resized)

            txt_filename = "Data_0/vectors_0.txt"
            with open(txt_filename, 'a') as f:
                np.savetxt(f, [roi_vector], fmt='%d', delimiter=',')

            label_int = label_int_var.get()
            label_str = label_str_var.get()

            if label_int.isdigit() and label_str:
                with open("Data_0/labels_int.txt", 'a') as f:
                    f.write(f"{label_int}\n")

                with open("Data_0/labels_string.txt", 'a') as f:
                    f.write(f"{label_str}\n")

                print(f"Imagen y datos guardados: {img_filename}, {txt_filename}, {label_int}, {label_str}")
                data_counter += 1
            else:
                messagebox.showerror("Error", "Etiqueta numérica debe ser un entero y la etiqueta de texto no puede estar vacía.")

def quit_program():
    root.quit()
    cap.release()
    cv2.destroyAllWindows()

# Configuración de tkinter
root = tk.Tk()
root.title("Captura de Datos")

label_int_var = tk.StringVar()
label_str_var = tk.StringVar()

# Crear botones y campos de entrada
capture_button = tk.Button(root, text="Capturar Imagen", command=capture_image)
capture_button.pack()

quit_button = tk.Button(root, text="Salir", command=quit_program)
quit_button.pack()

label_int_entry = tk.Entry(root, textvariable=label_int_var)
label_int_entry.pack()
label_int_label = tk.Label(root, text="Etiqueta Numérica (int)")
label_int_label.pack()

label_str_entry = tk.Entry(root, textvariable=label_str_var)
label_str_entry.pack()
label_str_label = tk.Label(root, text="Etiqueta de Texto (string)")
label_str_label.pack()

# Ejecutar el bucle de tkinter
def tkinter_loop():
    global frame
    while True:
        success, frame = cap.read()
        frame = rotateImage(frame, 0, scale=1,
                                       keepSize=True)
        if not success:
            break

        frame_copy = frame.copy()

        if bbox is not None:
            drawBox(frame, bbox)
            drawBox(frame_copy, bbox)

            x, y, w, h = bbox
            roi = frame[int(y):int(y+h), int(x):int(x+w)]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (30, 30))
                cv2.imshow("ROI", roi_resized)

        cv2.imshow("Camera", frame)
        cv2.imshow("Image", frame_copy)

        root.update_idletasks()
        root.update()

tkinter_loop()
