import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('model.h5')

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        
        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces=faceDetect.detectMultiScale(gray_img, 1.3, 5)
        
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            
            #find max indexed array
            max_index = np.argmax(predictions[0])
            
            emotions = ['Stress','Neutral' , 'Stress', 'Not Stress', 'Stress', 'Not Stress']
            predicted_emotion = emotions[max_index]
            if predicted_emotion =='Stress':
                #print('stress')
                #cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                i = 0
                while i <200:
                    cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    resized_img = cv2.resize(frame, (1000, 700))
                    cv2.imshow('Facial emotion analysis ',resized_img)
                    i = i+1
                    #print(i)
            elif predicted_emotion =='Neutral':
                #print('Neutral')
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                resized_img = cv2.resize(frame, (1000, 700))
                cv2.imshow('Facial emotion analysis ',resized_img)
                predictions = model.predict(img_pixels)
            else:
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                resized_img = cv2.resize(frame, (1000, 700))
                cv2.imshow('Facial emotion analysis ',resized_img)
                predictions = model.predict(img_pixels)
        #if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        #    break
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()