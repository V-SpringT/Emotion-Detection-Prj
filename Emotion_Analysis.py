import logging
import queue
import os.path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode
import matplotlib.pyplot as plt

def webcam_detect():
    #Create path weight
    relative_path = r"Model\weight_face_emotion_1.pt"
    defaul_dir = os.path.dirname(__file__)
    file_path_weight = os.path.join(defaul_dir, relative_path)

    logger = logging.getLogger(__name__)

    #load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Deep_Emotion()
    net.load_state_dict(torch.load(file_path_weight, map_location=device))
    net.to(device)

    webrtc_ctx = webrtc_streamer(
        key="video-sendonly",
        mode=WebRtcMode.SENDONLY,
        # rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True},
    )
    img_container = {"Angry":0, "Disgust":0, "Fear":0, "Happy":0, "Sad":0, "Surprise":0, "Neutral":0}

    #create chart place
    img_place = st.empty()
    txt_place = st.empty()
    fig_place = st.empty()
    fig, ax = plt.subplots()


    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            opencv_image = video_frame.to_ndarray(format="rgb24")
            
            #face detection
            faceCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 6, minSize=(50,50))
            
            if len(faces) != 0:
                # get largest rectangle
                areas = [w*h for x, y, w, h in faces]
                i_biggest = np.argmax(areas)
                biggest_face = faces[i_biggest]
                x, y, w, h = biggest_face

                # process model input and get its output
                roi_color = opencv_image[y: y + h, x:x + w]
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                graytemp = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                final_image = cv2.resize(graytemp, (48, 48))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image/255.0
                input_feature = torch.from_numpy(final_image)
                input_feature = input_feature.type(torch.FloatTensor)
                input_feature = input_feature.to(device)
                outputs = net(input_feature)
                pred = F.softmax(outputs, dim=1)
               

                # put output on frame
                font = cv2.FONT_HERSHEY_PLAIN
                status = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
                cv2.putText(opencv_image, status[int(torch.argmax(pred))] , (x - 10, y - 20), font, 2, (0, 255, 0), 2, cv2.LINE_4)
                img_container[status[int(torch.argmax(pred))]]+=1

                #show frame
                img_place.image(opencv_image)
                #Create percent emotion frequencies
                total = sum(img_container.values())
                per = {key: (value / total) * 100 for key, value in img_container.items()}
                categories = list(per.keys())
                values = list(per.values())
                ax.clear()
                ax.set_xlim(0, 100)
                bars = ax.barh(categories, values) 
                txt_place.title("Emotion Chart")
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                                textcoords='offset points', ha='left', va='center')
                #show chart
                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break