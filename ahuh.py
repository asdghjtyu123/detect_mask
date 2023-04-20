import mediapipe as mp
import cv2

# from cv2 import cv2

import numpy as np

import tensorflow as tf

           
def get_bounding_rectangle(image,region, color, radius):
    
    height, width = image.shape[0:2]
    
    bounding_box = region.location_data.relative_bounding_box   
                 
    left = bounding_box.xmin
    top = bounding_box.ymin
    
    right = left + bounding_box.width
    bottom = top + bounding_box.height
    
    left = (int)(left * width)
    top = (int)(top * height)
    
    if left <0:
        left =0
    if top <0:
        top = 0
    
    right = (int)(right * width)
    bottom = (int)(bottom * height)
    

    
    return left, top, right, bottom


def drw_bounding_rectangle(left, top, right, bottom, image, color, str_Mask, str_Score):
    
    cv2.rectangle(image, (left, top), (right, bottom), color, 1)
    
    cv2.line(image, (left, top), (left, top+10), color, 3)
    cv2.line(image, (left, top), (left+10, top), color, 3)
    
    cv2.line(image, (right, top), (right, top+10), color, 3)
    cv2.line(image, (right, top), (right-10, top), color, 3)
    
    
    cv2.line(image, (left, bottom), (left, bottom-10), color, 3)
    cv2.line(image, (left, bottom), (left+10, bottom), color, 3)
    
    cv2.line(image, (right, bottom), (right, bottom-10), color, 3)
    cv2.line(image, (right, bottom), (right-10, bottom), color, 3)
    
    cv2.putText(image, str_Mask, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  color,  2)
    
    cv2.putText(image, str_Score, (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  color,  2)
    
    
def convert_img_np(image_resize, target_size):
    
    image_s = np.array(image_resize, dtype=object)    
    image_s = image_s.reshape(1, target_size, target_size, 3)
    image_s = image_s.astype('float32')     
    image_s /= 255
    
    return image_s
def preprocessing(img):
    img=img.astype("uint8")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img


COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)


face_detection = mp.solutions.face_detection

capture = cv2.VideoCapture(0)


MODEL_SELECTION = 1
CONFIDENCE = 0.5

TARGET_SIZE = 256

detection = face_detection.FaceDetection(model_selection=MODEL_SELECTION, 
                                         min_detection_confidence= CONFIDENCE)

model_path = "/home/hoang/Downloads/detect_mask-main/MyTrainingModel.h5"
model = tf.keras.models.load_model(model_path)


while True:
    results, image = capture.read()
    
    if results:
        
        image_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        outputs = detection.process(image_convert)

        if outputs.detections:

             for region in outputs.detections:   
                                
                left, top, right, bottom = get_bounding_rectangle(image,region, COLOR_GREEN, 2)
                
                crop_image = image[top:bottom, left:right]
            
                           
                image_resize = cv2.resize(crop_image, (32, 32))

                image_s = preprocessing(image_resize)
                image_s=image_s.reshape(1, 32, 32, 1)
                # detect_class = model.predict_classes(image_s)
                
                r = model.predict(image_s)    
                print("*"*10)
                a = np.max(r)
                detect_class=np.argmax(r,axis=1)
                class_index = np.where(r==a)
                
                detect_class = class_index[1]

                score = r[0][0]              
                score = float("{:.2f}".format(score))
                
                if r[0][1] > 0.7:
                    color = COLOR_GREEN
                    str_Mask = 'WITH MASK'
                else:
                    color = COLOR_RED
                    str_Mask = 'NO MASK'
                
                drw_bounding_rectangle(left, top, right, bottom, image, color, str_Mask, str(score))

                
                
        cv2.imshow("Face_Detection", image)     
       
        if cv2.waitKey(1) & 255 == 27:
           break
        
capture.release()
cv2.destroyAllWindows()