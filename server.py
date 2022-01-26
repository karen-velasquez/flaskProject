import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from flask import Flask

'''Las librerias'''
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import pandas as pd
import pickle
import csv
#importando la precision de prediccion
from sklearn.metrics import accuracy_score

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def gen(filename): 
    #abrimos el modelo guardado
    #el segundo parametro es read binary
    with open('prediction_wushu.pkl','rb') as f:
        model=pickle.load(f)
    mp_drawing=mp.solutions.drawing_utils #drawing helpers
    #se obtiene el modelo 
    mp_holistic=mp.solutions.holistic  #mediapipe solutions
    #3. APLICANDO ESTILO Y CAMBIO DE COLORES AL MODELO
    #1. GET REALTIME WEBCAM FEED------------------------------
    #capturamos el dispositivo y le pasamos el numero de dispositivo del webcam
    #VIDEO FEED
    cap=cv2.VideoCapture('static/uploads/'+filename)
    #nombre='wushu2.mp4'
    # Initialize the VideoCapture object to read from a video stored in the disk.
    #cap = cv2.VideoCapture('myvideo/examples/'+nombre)

    #Para grabar el videooooooooooooooooooooooo
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    frame_counter = 0

    # Set video camera size
    cap.set(3,1280)
    cap.set(4,960)

    #Grabando video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    
    
    #LOGOOOOOOOOOOOOOOOO
    # Read logo and resize
    logo = cv2.imread('uie1.jpg')
    size = 400
    logo = cv2.resize(logo, (size, size))
    # Create a mask of logo
    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
 
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #GUARDAR en mp4 

    #out = cv2.VideoWriter('wushubasic/mayorporcentaje/'+nombre,cv2.VideoWriter_fourcc('m','p','4','v'), 10, (frame_width,frame_height))
    #out = cv2.VideoWriter('myvideo/videos/'+nombre,cv2.VideoWriter_fourcc('m','p','4','v'), 10, (frame_width,frame_height))
    #------------------------------------------


    #estamos configurnado e ingresando a nuestro modelo holistico
    #se especifica qeu tan buena es la deteccion y seguimineto y 
    #todo lo ponemos en holistic para trabajar con ello 
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        # mientras nuesro dispositovo de caputra este abierto 
        while cap.isOpened():
            #leyendo el feed de la camara
            ret,frame=cap.read();
            
            if ret==True:
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
                #Se cambio el llamado de FACE_CONNECTIONS a facemesh_contourS, FACEMESH_TESSELATI      
                #POSE DETECTION
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=4)
                                        ,mp_drawing.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2))



                #Export coordinates
                #Colocando todos los datos en un array y que esten en la misma linea
                try:
                    #Almacenamos todos los resultados de pose de landmarks en una variable
                    #Para el array obtenemos cada valor usamos un for para ir por cada uno y usamos flatten para poner
                    #los valores en una sola cadena
                    #si se quita flatten tendra varias matrices en una pequenia matriz
                    pose=results.pose_landmarks.landmark
                    pose_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in pose]).flatten())

                    row=pose_row
                #  row.insert(0,class_name)

                    #es practicamente el mismo codigo que arriba solo que para adicionar y no crear otro
                    #se pone a
                #   with open('coordenadass.csv',mode='a',newline='') as f:
                #      csv_writer=csv.writer(f,delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                #      csv_writer.writerow(row)
                    #Haciendo las detecciones
                    #ponemos el row que es el conjunto de datos en una variable X en formato frame
                    X=pd.DataFrame([row])
                    #Obtenemos el valor 0 de este frame y hacemos como arriba la prediccion
                    #este valor X es la concatenacion sin el label de clase
                    body_language_class=model.predict(X)[0]
                    body_language_prob=model.predict_proba(X)[0]
                # print(body_language_class, body_language_prob)

                    #Grab ear coords
                    #Estamos tomando las coordenadas de la oreja ya que ahi pondremos
                    #la clase
                    #en la primera parte obtenemos los datos y lo pasamos a una matriz numpy
                    #luego lo multiplicamos por las dimenciones de la camara para que sea ajuste a esto
                    #pero debemos pasar enteros para esto asi que se lo convierte con astype
                    coords=tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                                  results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y,
                                                  )), [frame_width,frame_height]).astype(int))


                    #renderizando la imagen
                    #Primero dibujamos un rectangulo entonces tendremos una superposicion de fondo de nuestra clase
                    #y lo demas es practicamente jugar con el posicionamiento, aqui agarramos los valores de la oreja puestos ante
                    #riormente en coords
                    cv2.rectangle(image, (coords[0],coords[1]+5),(coords[0]+len(body_language_class)*20, coords[1]-30),
                                (245,117,16),-1)
                    #escribiendo el texto
                    cv2.putText(image, body_language_class, coords,
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2, cv2.LINE_AA)

                    #Get status box
                    cv2.rectangle(image,(0,0),(450,120),(245,117,16),-1)

                    #Display class
                    cv2.putText(image,'CLASS'
                    ,(180,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
                    cv2.putText(image,body_language_class.split(' ')[0]
                            ,(180,80),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)


                    #Display class
                    cv2.putText(image,'PROB'
                    ,(15,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
                    cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            ,(10,80),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)


           
                    # Region of Image (ROI), where we want to insert logo
                    roi = image[-size-10:-10, -size-10:-10]

                    # Set an index of where the mask is
                    roi[np.where(mask)] = 0
                    roi += logo

                except:
                    pass


                #escribiendo el video 
                #out.write(image)
                #ahorita le decimos a nuestra cv2 que solo muestra la imagen del camara web
                #y cambiamos de frame a image ya que queremos enviarle nuestra imagen renderizada o dibujada
                #cv2.imshow('Raw Webcam Feed', image)
                image = cv2.resize(image, (0,0), fx=0.38, fy=0.38) 
                frame = cv2.imencode('.jpg', image)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
                #ahorita le decimos a nuestra cv2 que solo muestra la imagen del camara web
                #y cambiamos de frame a image ya que queremos enviarle nuestra imagen renderizada o dibujada
                #cv2.imshow('Raw Webcam Feed', image)

                #para salir de la camara web cuando se escrib
                if cv2.waitKey(10) & 0xFF==ord('q'):
                    break
            else:
                break

                
                
                
                
                #escribiendo el video 
                #out.write(image)
                #ahorita le decimos a nuestra cv2 que solo muestra la imagen del camara web
                #y cambiamos de frame a image ya que queremos enviarle nuestra imagen renderizada o dibujada
                #cv2.imshow('Raw Webcam Feed', image)

@app.route('/')
def upload_form():
	return render_template("pruebacss.html")


@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		return render_template("pruebacss.html", filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
    
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route("/video_feed/<filename>")
def video_feed(filename):
    return Response(gen(filename),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)
    


#<source src="{{ url_for('display_video', filename=filename) }}" type="video/mp4"></source>
'''
<div style="margin: 10px auto;">
        
		
	</div>
<video autoplay="autoplay" controls="controls" preload="preload">
            <source src="{{ url_for('video_feed', filename=filename) }}" type="video/mp4"></source>
		</video>'''