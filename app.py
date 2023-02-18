from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import numpy as np
import base64
import cv2
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
import statistics as st

app = Flask(__name__)


face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')

emotion_labels = ['angry','disgust','fear','happy','neutral', 'sad', 'surprise']

final_emotion=""
output=[]
def generate_frames(camera):
    while True:
        success, frame = camera.read()  # Read the frame from the video stream
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def cal(frame):
    j=0
    
    while (j<1):
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                max_index = np.argmax(label)
                output.append(label)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)       
        j=j+1


def final_call():
    arr = np.array(output)
    global final_emotion
    final_emotion = st.mode(arr)    
    print("predicted emotion = ",final_emotion)

         



@app.route('/')
def index():
    return render_template('index.html')




    
@app.route('/about')
def about():
    return render_template("about.html") 

@app.route('/detect')
def detect():
    return render_template('dummy.html')
  

@app.route('/send_frames', methods=['POST'])
def send_frames():
    data = request.json
    frame = data['frame']
    # convert the data URL to a numpy array
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)


    # do something with the frame here
    cal(frame)
    return {'success': True}
    



@app.route('/video')
def video():
    return Response(generate_frames(cv2.VideoCapture(0)), mimetype='multipart/x-mixed-replace; boundary=frame')

    


@app.route('/emo')
def emo():
    global final_emotion
    final_call()
    return render_template('emotion.html', predicted_emotion = final_emotion) 
    # return render_template("emotion.html", predicted_emotion = final_emotion) 


@app.route('/happy')
def happy():
    return render_template("happy.html")        

@app.route('/sad')
def sad():
    return render_template("sad.html")    

@app.route('/angry')
def angry():
    return render_template("angry.html")    

@app.route('/neutral')
def neutral():
    return render_template("neutral.html")    

@app.route('/surprise')
def surprise():
    return render_template("surprise.html")                    

@app.route('/disgust')
def disgust():
    return render_template("disgust.html")    

@app.route('/fear')
def fear():
    return render_template("fear.html")        


if __name__=="__main__":
    app.run(host='0.0.0.0')                 
