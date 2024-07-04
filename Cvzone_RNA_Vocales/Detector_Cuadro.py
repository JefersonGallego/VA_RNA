import cv2

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(1)

# Opcional: establecer la resolución de captura de la cámara (si la cámara lo soporta)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 0 = 1280x720 1 y 2 = 640x480  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

while True:
    success, frame = cap.read()
    if not success:
        break

    # Si hay un cuadro delimitador dibujado, mostrarlo en la ventana "Camera"
    if bbox is not None:
        drawBox(frame, bbox)

        # Extraer la ROI usando las coordenadas del cuadro delimitador
        x, y, w, h = bbox
        roi = frame[int(y):int(y+h), int(x):int(x+w)]

        # Convertir la ROI a escala de grises y redimensionar a 50x50 píxeles
        if roi.size > 0:  # Asegurarse de que la ROI no esté vacía
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (20, 20))
            # Mostrar la ROI redimensionada en la ventana "ROI"
            cv2.imshow("ROI", roi_resized)

            # Mostrar los valores como un vector
            roi_vector = roi_resized.flatten()
            print("Valores de la ROI redimensionada como vector:")
            print(roi_vector)

    # Mostrar el frame de la cámara con el cuadro delimitador
    cv2.imshow("Camera", frame)

    # Si hay un cuadro delimitador dibujado, mostrarlo también en la ventana "Image"
    if bbox is not None:
        frame_copy = frame.copy()
        drawBox(frame_copy, bbox)
        cv2.imshow("Image", frame_copy)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
