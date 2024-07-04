from pyzbar import pyzbar
import cv2
import pandas as pd
import os

# Definimos los nombres de los archivos
txt_filename = 'barcode_data.txt'
xlsx_filename = 'barcode_data.xlsx'

# Si el archivo Excel ya existe, lo cargamos, de lo contrario creamos un nuevo DataFrame
if os.path.exists(xlsx_filename):
    df = pd.read_excel(xlsx_filename)
else:
    df = pd.DataFrame(columns=["Barcode Data"])

def main():
    try:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Could not open video device")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            barcodes = pyzbar.decode(frame)
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                text = "{} ({})".format(barcodeData, barcodeType)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Mostrar información en la terminal
                print(f"[INFO] Found {barcodeType} barcode: {barcodeData}")
                
                # Guardar en archivo de texto
                with open(txt_filename, 'a') as file:
                    file.write(f"{barcodeData}\n")
                
                # Agregar datos al DataFrame usando concat
                new_row = pd.DataFrame({"Barcode Data": [barcodeData]})
                global df
                df = pd.concat([df, new_row], ignore_index=True)
                
                # Guardar DataFrame en archivo Excel
                df.to_excel(xlsx_filename, index=False)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
