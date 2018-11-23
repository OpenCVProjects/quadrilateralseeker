# Autor: Jose Estevez Fernandez

import numpy as np
import cv2

# Inicia la captura de vídeo
video = cv2.VideoCapture(0)

# Bucle para procesar el video capturado
while True:

    check, frame = video.read()

    # Convertimos a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("suavizado", gris)

    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5,5), 0)

    cv2.imshow("suavizado", gauss)

    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 250)

    cv2.imshow("canny", canny)

    # Buscamos los contornos
    (_, contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cntr in contornos:
        if cv2.contourArea < 8000:
            continue

        approx = cv2.approxPolyDP(cntr,0.06*cv2.arcLength(cntr,True),True)
        if len(approx)==4:
            print(2)
            (x,y,w,h) = cv2.boundingRect(cntr)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3 )

    cv2.imshow("contornos", frame)

    key = cv2.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
