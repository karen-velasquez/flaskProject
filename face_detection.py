'''Las librerias'''
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import pandas as pd
import pickle
import csv
#configurando dos lines de media pipe para nuestro modelo
#primero se exporta las caracteristicas de dibujo
mp_drawing=mp.solutions.drawing_utils #drawing helpers
#se obtiene el modelo 
mp_holistic=mp.solutions.holistic  #mediapipe solutions

cap=cv2.VideoCapture(r'C:\Users\asus\Desktop\UNIVERSIDAD\PASANTIA\ANACONDA\PROYECTO\flask\corte7.mp4')
face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+
        "haarcascade_frontalface_default.xml")
#3. APLICANDO ESTILO Y CAMBIO DE COLORES AL MODELO
#1. GET REALTIME WEBCAM FEED------------------------------
#capturamos el dispositivo y le pasamos el numero de dispositivo del webcam
cap=cv2.VideoCapture('corte7.mp4')

#estamos configurnado e ingresando a nuestro modelo holistico
#se especifica qeu tan buena es la deteccion y seguimineto y 
#todo lo ponemos en holistic para trabajar con ello 
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    # mientras nuesro dispositovo de caputra este abierto 
    while cap.isOpened():
        #leyendo el feed de la camara
        ret,frame=cap.read();
        #Recoloreando de bgr a rgb para enviar los datos
        #de manera correcta
        #cuando lo usamos con opencv utilizamos bgr pero para enviarlo 
        #la modelo holistico requerimos rgb
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False

        #Realizando decisiones
        results=holistic.process(image)
      #  print(results.face_landmarks)
        
        #Comenzamos a devolverlo a su color para luego pintarlo encima de 
        #la imagen con opencv YA QUE SINO TIENE UNA COLORACION EXTRA;A
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Dibujando las marcas de la cara 
        #Se cambio el llamado de FACE_CONNECTIONS a facemesh_contourS, FACEMESH_TESSELATION
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                 mp_drawing.DrawingSpec(color=(80,110,10),thickness=2, circle_radius=1)
                                ,mp_drawing.DrawingSpec(color=(80,250,121),thickness=2, circle_radius=1))
        
        
        #RIGHT HAND
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10),thickness=2, circle_radius=4)
                                ,mp_drawing.DrawingSpec(color=(80,44,121),thickness=2, circle_radius=2))
      
        
        #LEFT HAND
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76),thickness=2, circle_radius=4)
                                ,mp_drawing.DrawingSpec(color=(121,44,250),thickness=2, circle_radius=2))
      
        
        #POSE DETECTION
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=4)
                                ,mp_drawing.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2))
      
        
        
        
         #ahorita le decimos a nuestra cv2 que solo muestra la imagen del camara web
        #y cambiamos de frame a image ya que queremos enviarle nuestra imagen renderizada o dibujada
        cv2.imshow('Raw Webcam Feed', image)
        
        #para salir de la camara web cuando se escrib
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()