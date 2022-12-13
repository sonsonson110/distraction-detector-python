import os
from keras.utils.image_utils import img_to_array
import cv2
from keras.models import load_model
import numpy as np
import datetime as dt
import time

#tao file log
date_today = dt.date.today()
log_txt = open(os.path.join(os.getcwd(), 'log', f'(huy) {date_today}.txt'), 'a+')
#thoi gian canh cao
time_noface = None
time_noface_flag = False
time_distract = None
time_distract_flag = False
# models
# face and eyes are templates from opencv
# distract model is a TF CNN model trained using Keras (see /src/cnn/train.py) 
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml'))
distract_model = load_model(os.path.join(os.getcwd(),'combine.hdf5'), compile=False)
# frame params
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

# Video writer
# - frame width and height must match output frame shape
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
output_name = f'(huy) {date_today}.avi'
path = os.path.join(os.getcwd(), 'log', output_name)
video_out = cv2.VideoWriter(path, fourcc, 24.0,(1280, 720))

# khởi tạo camera
cv2.namedWindow('Camera')
camera = cv2.VideoCapture(r'testing_huy.mp4') #có thể sử dụng video khác để phân tích bằng cách thay tham số bằng đường dẫn
#camera = cv2.VideoCapture(0) 

# kiểm tra cam
if (camera.isOpened() == False): 
    print("Khong the ket noi toi camera/media")

while True:
    # get frame
    ret, frame = camera.read()

    #hiển thị thời gian thực
    cv2.putText(frame, 
                f'{dt.datetime.now()}', 
                (50, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255), 
                1, 
                cv2.LINE_4)
    # if we have a frame, do stuff
    if ret:

        # make frame bigger
        frame = cv2.resize(frame, (1280, 720))

        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face(s)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)
        
        # for each face, detect eyes and distraction
        if len(faces) > 0:
            #reset thoi gian canh cao
            time_noface = None
            time_noface_flag = False
            # loop through faces
            for (x,y,w,h) in faces:
                # draw face rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # get gray face for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                # get colour face for distraction detection (model has 3 input channels - probably redundant)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # init probability list for each eye prediction
                probs = list()

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # get colour eye for distraction detection
                    roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                    # match CNN input shape
                    roi = cv2.resize(roi, (64, 64))
                    # normalize (as done in model training)
                    roi = roi.astype("float") / 255.0
                    # change to array
                    roi = img_to_array(roi)
                    # correct shape
                    roi = np.expand_dims(roi, axis=0)

                    # distraction classification/detection
                    prediction = distract_model.predict(roi)
                    # save eye result
                    probs.append(prediction[0])

                # get average score for all eyes
                probs_mean = np.mean(probs)

                # get label
                if probs_mean <= 0.5:
                    if time_distract == None:
                        time_distract = time.time()
                    label = 'mat tap trung'
                    if (time.time() - time_distract > 5) and not time_distract_flag:
                        time_distract_flag = True
                        log_txt.write(f'{dt.datetime.now()}: khong chu y qua 5s\n')
                else:
                    time_distract = None
                    time_distract_flag = False
                    label = 'tap trung'
                
                # insert label on frame
                cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 
                'CANH BAO KHONG CHU Y', 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
            #tang thoi gian canh cao
            if (time_noface == None):
                time_noface = time.time()
            #log neu vuot qua 10s
            if (time.time() - time_noface > 10 and not time_noface_flag):
                time_noface_flag = True
                log_txt.write(f'{dt.datetime.now()}: khong co mat qua 10s\n')

        # Write the frame to video
        video_out.write(frame)

        # display frame in window
        cv2.imshow('Camera', frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # no frame, don't do stuff
    else:
        break

# close
camera.release()
video_out.release()
log_txt.close()
cv2.destroyAllWindows()