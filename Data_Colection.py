import streamlit as st
from streamlit_webrtc import webrtc_streamer,WebRtcMode
from streamlit_session_memo import st_session_memo
from PIL import Image
import cv2
import av
from io import *
import os
import csv
import pandas as pd
import numpy as np

# sai ve thu tu
realEmo = None
imgSave2 = None
imgSave = None
emoUp = None
check=False
# tao dict
EmotionIdx = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
def webcam_input(selected_emotion):

    @st_session_memo
    def update_text():
        return selected_emotion


    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        global realEmo,imgSave2
        # 
        opencv_image = frame.to_ndarray(format="bgr24")
        txt = update_text()
        # Flip img 
        opencv_image = cv2.flip(opencv_image, flipCode=1)

        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(faces) != 0:
            # get largest rectangle
            areas = [w*h for x, y, w, h in faces]
            i_biggest = np.argmax(areas)
            biggest_face = faces[i_biggest]
            x, y, w, h = biggest_face

            # process model input and get its output
            roi_color = opencv_image[y: y + h, x:x + w]
            cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # put output on frame
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(opencv_image, txt , (x - 10, y - 20), font, 2, (0, 255, 0), 2, cv2.LINE_4)
            # Tao data train
            roi_color = opencv_image[y:y + h, x:x + w]
            imgSave2 = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            imgSave2 = cv2.resize(imgSave2, (48, 48)) 
            realEmo = txt
        return av.VideoFrame.from_ndarray(opencv_image, format="bgr24")

    ctx = webrtc_streamer(
        key="Data-Collection",
        video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV,
        # rtc_configuration={"iceServers": get_ice_servers()},
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
    # Create capture button
    if ctx:
        #Create path
        relative_path = r"Data\train.csv"
        defaul_dir = os.path.dirname(__file__)
        file_path = os.path.join(defaul_dir, relative_path)

        global check,imgSave,emoUp
        df = pd.read_csv(file_path)
        max_index = df['rank'].max()
        if(str(int(max_index)).isdigit()): max_index +=1
        else: max_index = 1
        # buttons position 
        left, mid, right = st.columns(3)
        with left:
            capBut = st.button("Capture")
        if capBut :
            img = Image.fromarray(imgSave2)
            st.image(img,use_column_width=True)
            check = True
            imgSave = imgSave2
            emoUp = realEmo
            capBut = False
        if check:
            #create save button
            global serial_captured
            img_array = np.array(imgSave)
            jpeg_image = cv2.imencode(".jpg", img_array)[1].tobytes()
            buffer = BytesIO()
            buffer.write(jpeg_image)
            with mid:
                btn = st.download_button(
                    label="Save",
                    data=buffer,
                    file_name=f"image{max_index}.png",
                    mime="image/png"
                )
            #create Update button
            with right:
                updateBut = st.button("Update")
            if updateBut:
                if not os.path.exists(file_path):
                    with open(file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(['rank','emotion', 'pixels'])
                else:
                    with open(file_path, 'a') as f:
                        writer = csv.writer(f)
                        flattened = imgSave.flatten()
                        pixel_values = ' '.join(map(str, flattened))
                        writer.writerow([max_index,EmotionIdx[emoUp], pixel_values])
            max_index+=1

            


    

