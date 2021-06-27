from operator import ge
import numpy as np
import cv2
import dlib
from pathlib import Path
import glob
import math
from PIL import Image
from tensorflow.keras.models import load_model
from architecture import * 
from tensorflow.keras.models import load_model
import tensorflow.keras
from utils import detector_utils as detector_utils
import tensorflow as tf
from keras.preprocessing import image

def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def find_face(face_detector=0, face_encoder=0, source=0, known_face_encodings=0):
    if isinstance(source, str):
        img = cv2.imread(source)
    else:
        img = source
    detected_faces = face_detector(img, 1)
    encodings = []
    locations = []
    names = []
    Path("./faces").mkdir(exist_ok=True)
    if isinstance(source, str):
        print(f'find {len(detected_faces)} face in images.')
    for i, face_rect in enumerate(detected_faces):
        locations.append((face_rect.top(),face_rect.bottom(),face_rect.left(),face_rect.right()))
        face = img[face_rect.top():face_rect.bottom(),face_rect.left():face_rect.right(),:]
        if (face_rect.bottom()-face_rect.top())*(face_rect.right()-face_rect.left()) <=0 or face_rect.top() < 0 or face_rect.left() < 0 or face_rect.bottom() >= img.shape[0] or face_rect.right() >= img.shape[1]:
            continue
        face = cv2.resize(face, (160,160))
        #------ align face using eyes ------
        face_raw = face.copy()
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(face_gray)

        index = 0
        if (len(eyes)==2):
            try:
                for (eye_x, eye_y, eye_w, eye_h) in eyes:
                   if index == 0:
                      eye_1 = (eye_x, eye_y, eye_w, eye_h)
                   elif index == 1:
                      eye_2 = (eye_x, eye_y, eye_w, eye_h)
                 
                   cv2.rectangle(face,(eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), (255, 255, 0), 2)
                   index = index + 1
                if eye_1[0] < eye_2[0]:
                   left_eye = eye_1
                   right_eye = eye_2
                else:
                   left_eye = eye_2
                   right_eye = eye_1
                left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
                 
                right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
                right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
                 
                cv2.circle(face, left_eye_center, 2, (255, 0, 0) , 2)
                cv2.circle(face, right_eye_center, 2, (255, 0, 0) , 2)
                cv2.line(face,right_eye_center, left_eye_center,(67,67,67),2)
                if left_eye_y > right_eye_y:
                   point_3rd = (right_eye_x, left_eye_y)
                   direction = -1 #rotate same direction to clock
                   #print("rotate to clock direction")
                else:
                   point_3rd = (left_eye_x, right_eye_y)
                   direction = 1 #rotate inverse direction of clock
                   #print("rotate to inverse clock direction") 
                cv2.circle(face, point_3rd, 2, (255, 0, 0) , 2)  
                cv2.line(face,right_eye_center, left_eye_center,(67,67,67),2)
                cv2.line(face,left_eye_center, point_3rd,(67,67,67),2)
                cv2.line(face,right_eye_center, point_3rd,(67,67,67),2)

                a = euclidean_distance(left_eye_center, point_3rd)
                b = euclidean_distance(right_eye_center, left_eye_center)
                c = euclidean_distance(right_eye_center, point_3rd)

                cos_a = (b*b + c*c - a*a)/(2*b*c)
                angle = np.arccos(cos_a)
                angle = (angle * 180) / math.pi

                if direction == -1:
                    angle = 90 - angle

                new_img = Image.fromarray(face_raw)
                face = np.array(new_img.rotate(direction * angle))
            except Exception:
                pass

        face = face[:60,:,:]
        face = cv2.resize(face, (160,160))
        if isinstance(source, str):
            cv2.imwrite(f"faces/{len(known_face_encodings)}.jpg", face)
            name = path.replace('imgs/', '')
            name = name.replace('.jpg', '')
            names.append(name)
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        samples = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(samples)[0]
        encodings.append(encode)
    return names, locations, encodings

def cal_dist(tar_encoding, known_face_encodings):
    known_face_encodings = np.array(known_face_encodings)
    N = known_face_encodings.shape[0]
    tar_encoding = np.tile(tar_encoding, (N, 1))
    dist = np.linalg.norm(tar_encoding - known_face_encodings, axis=1)
    return dist

def faceLocationValid(face_location, im_width, im_height):
    (top,bottom,left,right) = (face_location[0]*4,face_location[1]*4,face_location[2]*4,face_location[3]*4)
    if top < 0 or bottom >= im_height or left < 0 or right >= im_width:
        return False
    return True


# ------ load face detector & face encoder model ------
face_detector = dlib.get_frontal_face_detector()
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

# ------ load gesture recognition model ------
detection_graph, sess = detector_utils.load_inference_graph()
gesture_recognizer = load_model('model_gesture_gene_OK.h5')

# ------ load mask detection model ------
detection_graph, sess = detector_utils.load_inference_graph()
mask_detector = load_model('model_mask.h5')

# ------ Load sample pictures and learn their encodings & record their name ------
known_face_encodings = []
known_face_names = []
for path in glob.glob("imgs/*.jpg"):
    print('processing '+path)
    names, _, encodings = find_face(face_detector=face_detector, face_encoder=face_encoder, source=path, known_face_encodings=known_face_encodings)
    known_face_encodings = known_face_encodings + encodings
    known_face_names= known_face_names + names

# ------ Initialize some variables ------
face_locations = []
face_encodings = []
ask_names = []

idx_cnt = 0
best_match_index = np.zeros(1)
build_encoding = 0
build_flag = False
face_threshold = 5

cap=cv2.VideoCapture(0)
im_width, im_height = (int(cap.get(3)), int(cap.get(4)))
prediction = [[0,0]]
(left, right, top, bottom) = (0,0,0,0)
draw = False
gesture_threshold = 0.9
data = np.ndarray(shape=(1, 160, 160, 1), dtype=np.float32)
last_ges = 0

mode = "checkFace"
status = "bad"
gesture = "yes"
no_ges_cnt = 0
asked_cnt = 0

MASK = False

process_this_frame = True
cnt = 0
last_idx = -1
accu_dist = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if process_this_frame:
        # --- Find all the faces and face encodings ---
        _, face_locations, face_encodings = find_face(face_detector=face_detector, face_encoder=face_encoder, source=small_frame, known_face_encodings=known_face_encodings)
        if len(face_locations) > 0 and faceLocationValid(face_locations[0], im_width, im_height):
            face_img = frame[face_locations[0][0]*4:face_locations[0][1]*4,face_locations[0][2]*4:face_locations[0][3]*4]
            cv2.imwrite('temp.jpg',face_img)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mask_detector.predict(test_image)[0][0]
            if pred == 0:
                MASK = True
        if mode == 'checkFace':
            if len(face_locations) > 0 and faceLocationValid(face_locations[0], im_width, im_height):
                face_distances = cal_dist(face_encodings[0], known_face_encodings)
                confidence_list = []
                for i in range(len(face_distances)):
                    confidence = 0
                    if 1-0.1*face_distances[i] > 0:
                        confidence = 1-0.1*face_distances[i]
                    else:
                        confidence = 0
                    confidence_list.append(round(confidence,3))
                #print(confidence_list)
                print(face_distances)
                best_match_index = np.argsort(face_distances)

                if last_idx != best_match_index[0]:
                    cnt = 0
                    accu_dist = 0
                    last_idx = best_match_index[0]
                else:
                    cnt += 1
                    accu_dist += face_distances[best_match_index[0]]
            else:
                cnt = 0
                accu_dist = 0
                last_idx = -1
            if cnt > 10:
                if accu_dist/cnt < face_threshold:
                    mode = 'checkGesture'
                else:
                    mode = 'finish'
                    status = 'bad'
                cnt = 0
                accu_dist = 0
                ask_name = known_face_names[best_match_index[0]]
        if mode == 'checkGesture':
            no_ges_cnt = 0
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
            draw = False
            (l,r,t,b) = (0,0,0,0)
            if len(face_locations) > 0 and faceLocationValid(face_locations[0], im_width, im_height):
                (t,b,l,r) = (face_locations[0][0]*4,face_locations[0][1]*4,face_locations[0][2]*4,face_locations[0][3]*4)
            (left, right, top, bottom) = (0,0,0,0)
            score = 0
            (left, right, top, bottom) = (int(boxes[0][1] * im_width), int(boxes[0][3] * im_width),int(boxes[0][0] * im_height), int(boxes[0][2] * im_height))
            center_i = (top+bottom)//2
            center_j = (left+right)//2
            for i in range(boxes.shape[0]):
                (left, right, top, bottom) = (int(boxes[i][1] * im_width), int(boxes[i][3] * im_width),int(boxes[i][0] * im_height), int(boxes[i][2] * im_height))
                score = scores[i]
                center_i = (top+bottom)//2
                center_j = (left+right)//2
                err = 0
                if (center_i > t and center_i < b):
                    err += 1
                if (center_j > l and center_j < r):
                    err += 1
                if err <= 1:
                    break
            if (score > gesture_threshold):
                no_ges_cnt = 0
                draw = True
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
                (_, cr, _) = cv2.split(ycrcb)
                cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
                _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                temp = skin[top:bottom,left:right]
                cv2.imwrite('temp.jpg',temp)
                test_image=image.load_img('temp.jpg',target_size=(160,160,1), grayscale=True)
                test_image=image.img_to_array(test_image)
                test_image=np.expand_dims(test_image,axis=0)
                pred=gesture_recognizer.predict(test_image)[0][0]
                if pred == 1:
                    gesture = 'YES'
                else:
                    gesture = 'NO'
                if last_ges == gesture:
                    cnt += 1
                else:
                    cnt = 0
                    last_ges = gesture
                if cnt > 10:
                    cnt = 0
                    if gesture == 'YES':
                        mode = 'finish'
                        status = 'good'
                    elif gesture == 'NO':
                        mode = 'finish'
                        status = 'bad'
            else:
                no_ges_cnt += 1
                cnt = 0
                if no_ges_cnt > 100:
                    mode == 'checkFace'
    
    process_this_frame = not process_this_frame

    # Display the results
    for (t, b, l, r) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        if faceLocationValid((t,b,l,r), im_width, im_height) == False:
            continue
        cv2.rectangle(frame, (l*4, t*4), (r*4, b*4), (0, 0, 255), 2)

    if mode == 'checkFace':
        cv2.rectangle(frame, (0, 0), (im_width, 35), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = "PLEASE FACE THE CAMERA"
        cv2.putText(frame, text, (10, 30), font, 1.0, (255, 255, 255), 1)
    if mode == 'checkGesture':
        if len(face_locations) > 0 and faceLocationValid(face_locations[0], im_width, im_height):
            cv2.rectangle(frame, (face_locations[0][2]*4, face_locations[0][1]*4 - 35), (face_locations[0][3]*4, face_locations[0][1]*4), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = "ARE YOU "+ ask_name+"?"
            cv2.putText(frame, text, (face_locations[0][2]*4 + 6, face_locations[0][1]*4 - 6), font, 1.0, (255, 255, 255), 1)
        if draw == True:
            center = ((left+right)//2,(top+bottom-35)//2)
            radius = (bottom-35-top)//2
            cv2.ellipse(frame,center,(radius,radius),0,-90,-90+36*cnt,(255,255,0),40)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, gesture, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    if mode == 'finish':
        tmp = ""
        if MASK == False:
            tmp = " AND PLEASE WEAR A FACE MASK BEFORE ENTERING."
        if status == 'good':
            cv2.rectangle(frame, (0, im_height-35), (im_width, im_height), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = "PASS." + tmp
            cv2.putText(frame, text, (6, im_height-6), font, 1.0, (255, 255, 255), 1)
        elif status == 'bad':
            cv2.rectangle(frame, (0, im_height-35), (im_width, im_height), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = "SORRY, NOT ALLOWED."
            cv2.putText(frame, text, (6, im_height-6), font, 1.0, (255, 255, 255), 1)
        cnt += 1
        if cnt > 70:
            cnt = 0
            mode = 'checkFace'
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()