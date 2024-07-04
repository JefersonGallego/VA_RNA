import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from keras import optimizers, losses, metrics
from cvzone.Utils import rotateImage

class SequentialClassifier:
    def __init__(self, model_json_path=None, weights_path=None, num_classes=None, input_shape=(30, 30)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

        if model_json_path and weights_path:
            self.load_model(model_json_path, weights_path)

    def load_model(self, model_json_path, weights_path):
        # Cargar modelo desde archivo JSON
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)

        # Cargar pesos al modelo
        self.model.load_weights(weights_path)

        # Compilar el modelo después de cargar los pesos
        self.model.compile(
            loss=losses.sparse_categorical_crossentropy,  # Función de pérdida
            optimizer=optimizers.Adam(learning_rate=0.0001),  # Optimizador Adam con tasa de aprendizaje especificada
            metrics=[metrics.SparseCategoricalAccuracy(), 'accuracy'],  # Métricas para evaluar el modelo
        )

        # Obtener el número de clases del modelo cargado si no se especificó previamente
        if not self.num_classes:
            self.num_classes = self.model.layers[-1].output_shape[-1]

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, self.input_shape)
        image_array = np.asarray(img_resized).astype(np.float32) / 255.0
        flattened_image = image_array.flatten()  # Aplanar la imagen a un vector unidimensional
        return np.expand_dims(flattened_image, axis=0)

    def get_prediction(self, img):
        input_data = self.preprocess_image(img)
        prediction = self.model.predict(input_data)
        index_val = np.argmax(prediction)
        return index_val

# Definir etiquetas de las clases
class_labels = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U', 5: 'Malo', 6: 'No Data'}

# Definir rutas a los archivos del modelo y el número de clases
model_json_path = 'modelo_30.json'
model_weights_path = 'modelo_pesos_30.weights.h5'
num_classes = 7 

# Iniciar captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Opcional: establecer la resolución de captura de la cámara (si la cámara lo soporta)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 0 = 1280x720 1 y 2 = 640x480  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Crear una ventana para la cámara y otra para mostrar el cuadro delimitador
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)  # Nueva ventana para mostrar la ROI

# Establecer el tamaño de la ventana de la imagen
cv2.resizeWindow("Image", 1280, 720)  # Ajustar según sea necesario
#cv2.resizeWindow("Image", 1280, 720)
# Función para dibujar el cuadro delimitador
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
    cv2.putText(img, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

bbox = None
drawing = False
label = ""  # Inicializar la variable label

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

classifier = SequentialClassifier(model_json_path, model_weights_path, num_classes=num_classes, input_shape=(30, 30))

while True:
    success, frame = cap.read()
    #frame = rotateImage(frame, 0, scale=1, keepSize=True)
    if not success:
        break

    # Si hay un cuadro delimitador dibujado, mostrarlo en la ventana "Camera"
    if bbox is not None:
        drawBox(frame, bbox)

        # Extraer la ROI usando las coordenadas del cuadro delimitador
        x, y, w, h = bbox
        roi = frame[int(y):int(y+h), int(x):int(x+w)]

        # Convertir la ROI a escala de grises y redimensionar a 20x20 píxeles
        if roi.size > 0:  # Asegurarse de que la ROI no esté vacía
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (30, 30))
            # Mostrar la ROI redimensionada en la ventana "ROI"
            cv2.imshow("ROI", roi_resized)

            # Realizar la predicción usando el modelo cargado
            index = classifier.get_prediction(roi_resized)
            label = class_labels[index]

            # Mostrar la etiqueta en la ventana "Camera"
            cv2.putText(frame, label, (x + 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el frame de la cámara con el cuadro delimitador
    cv2.imshow("Camera", frame)

    # Si hay un cuadro delimitador dibujado, mostrarlo también en la ventana "Image"
    if bbox is not None:
        frame_copy = frame.copy()
        drawBox(frame_copy, bbox)
        cv2.putText(frame_copy, label, (x + 100 , y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Image", frame_copy)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y destruir todas las ventanas
cap.release()
cv2.destroyAllWindows()
