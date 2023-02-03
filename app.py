from flask import Flask,render_template,Response
import cv2
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import statistics as st

face_classifier = cv2.CascadeClassifier(r'C:\Users\Diksha\OneDrive\Desktop\ML\Melodify_new\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Diksha\OneDrive\Desktop\ML\Melodify_new\model.h5')

emotion_labels = ['angry','disgust','fear','happy','neutral', 'sad', 'surprise']

app=Flask(__name__)
camera=cv2.VideoCapture(0)


final_emotion = ""
def recog(frame):
    i=0
    output=[]
    while (i<=0):
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
        i = i+1        
        key = cv2.waitKey(1)
        if key == 12: 
                camera.release()
                cv2.destroyAllWindows()
                break

    arr = np.array(output)
    global final_emotion
    final_emotion = st.mode(arr)
        

def generate_frames():
    j=0
    while (j<=30):
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:

            recog(frame)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()


        j=j+1
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    print("predicted emotion = ",final_emotion)  

    return render_template('emotion.html')      



@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template("about.html") 

@app.route('/detect')
def detect():
    return render_template('index2.html', predicted_emotion = final_emotion)
  

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emo')
def emo():
    return render_template("emotion.html", predicted_emotion = final_emotion) 

 

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
    app.run(debug=True)                   