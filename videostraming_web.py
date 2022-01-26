
import time
import cv2 
from flask import Flask, render_template, Response

app=Flask(__name__)


cap=cv2.VideoCapture(r'C:\Users\asus\Desktop\UNIVERSIDAD\PASANTIA\ANACONDA\PROYECTO\flask\corte7.mp4')
face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+
        "haarcascade_frontalface_default.xml")

def generate():
    while True:
        ret, frame=cap.read()
        if ret:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0),2)
            (flag,encodedImage)=cv2.imencode(".jpg",frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+
                bytearray(encodedImage)+b'\r\n')
            
            if cv2.waitKey(1) & 0xFF==ord("q"):
                break
        
'''cap=cv2.VideoCapture(r'C:\Users\asus\Desktop\UNIVERSIDAD\PASANTIA\ANACONDA\PROYECTO\flask\corte7.mp4')
    while (cap.isOpened()):
        ret, frame=cap.read()
        if ret==True:
            #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            (flag,encodedImage)=cv2.imencode(".jpg",frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+
                bytearray(encodedImage)+b'\r\n')
            time.sleep(0.1)
        else:
            break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0),2)
            (flag,encodedImage)=cv2.imencode(".jpg",frame)
            if not flag:
                continue
            yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+
                bytearray(encodedImage)+b'\r\n')
'''


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    app.run(debug=False)
    

