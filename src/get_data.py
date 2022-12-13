import os
import imutils
import cv2

# models
# face and eyes are templates from opencv
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml'))
# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

# image iterators
# i = số thứ tự ảnh được lấy ra
# j = kiểm soát lưu ảnh
i = 0
j = 0

# khởi tạo cửa sổ cam
cv2.namedWindow('Get data')
camera = cv2.VideoCapture(0) #lấy từ camera laptop

# Kiểm tra nếu có lỗi từ camera
if (camera.isOpened() == False): 
    print("Unable to read camera feed")

# liên tục lấy mẫu
while True:
    # đọc frame hiện tại trên màn hình
    ret, frame = camera.read()

    #đọc được frames
    if ret:
        
        # chỉnh size frame
        frame = imutils.resize(frame,width=frame_w)

        # sử dụng phương thức 'grayscale' cho tốc độ xử lý nhanh hơn
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # nhận diện khuân mặt
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)

        # với mỗi khuôn mặt, nhận diện mắt
        if len(faces) > 0:
            # loop through faces
            for (x,y,w,h) in faces:
                # vẽ các khung lên vật nhận diện
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                #tiền xử lý khuôn mặt để lấy mẫu mắt
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                # nhận diện mắt
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # duyệt qua từng mắt
                for (ex,ey,ew,eh) in eyes:
                    # vẽ các ô lên vật thể nhận diện
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # theo dõi các ô vuông
                    j += 1
                    # không lấy dữ liệu nếu ghi nhận 1 mắt
                    if j%2 == 0:
                        # tăng biến tên ảnh
                        i += 1
                        # tạo thông tin lưu trữ data
                        imgname = 'eye_('+str(i)+').jpg'
                        #imgname = 'eye_d('+str(i)+').jpg'
                        #nhat - son thay doi linh hoat khi lay du lieu
                        path = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'focus', imgname)
                        #path = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'distract', imgname)
                        print(f'File {imgname} created!')
                        # ghi dữ liệu vào ảnh
                        cv2.imwrite(path, roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w])

        # show frame in window
        cv2.imshow('Get data', frame)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# close
camera.release()
cv2.destroyAllWindows()
