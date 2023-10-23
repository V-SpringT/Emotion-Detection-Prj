import os.path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode
import av


def webcam_detect1():

    relative_path = r"Model\weight_face_emotion_1.pt"
    defaul_dir = os.path.dirname(__file__)
    file_path_weight = os.path.join(defaul_dir, relative_path)
    #load model
    device = torch.device('cpu')
    net = Deep_Emotion()
    net.load_state_dict(torch.load(file_path_weight, map_location=device))
    net.to(device)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        opencv_image = frame.to_ndarray(format="bgr24")

        # flip image
        opencv_image = cv2.flip(opencv_image, flipCode=1)
        
        #import face_detection
        faceCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 6, minSize=(50, 50))
        
        for x, y, w, h in faces:
            # draw retangle around the face
            roi_color = opencv_image[y: y + h, x:x + w]
            cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            face_roi = roi_color
            graytemp = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            final_image = cv2.resize(graytemp, (48, 48))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image/255.0
            input_feature = torch.from_numpy(final_image)
            input_feature = input_feature.type(torch.FloatTensor)
            input_feature = input_feature.to(device)
            outputs = net(input_feature)
            
            pred = F.softmax(outputs, dim=1)
            font = cv2.FONT_HERSHEY_PLAIN
            status = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            cv2.putText(opencv_image, status[int(torch.argmax(pred))], (x - 10, y - 20), font, 2, (0, 255, 0), 2, cv2.LINE_4)
        return av.VideoFrame.from_ndarray(opencv_image, format="bgr24") 
    

    ctx = webrtc_streamer(
            key="Data-Collection",
            video_frame_callback=video_frame_callback,
            mode=WebRtcMode.SENDRECV,
            # rtc_configuration={"iceServers": get_ice_servers()}, # deploy
            media_stream_constraints={
                "video": {"frameRate": {"ideal": 5}},
                "video": True, "audio": False
            },
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border": "5px yellow solid"},
                "controls": False,
                "autoPlay": True,
            },
        )

    # Xây dựng lược đồ thanh barh(matplotlib)