import streamlit as st
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from deep_emotion import Deep_Emotion
from PIL import Image
import csv
import pandas as pd

#Create path trainTrue
relative_path = r"Data\trainTrue.csv"
defaul_dir = os.path.dirname(__file__)
file_path_trainTrue = os.path.join(defaul_dir, relative_path)  

#Create path TrainFalse
relative_path = r"Data\trainFalse.csv"
defaul_dir = os.path.dirname(__file__)
file_path_trainFalse = os.path.join(defaul_dir, relative_path)   

# tao dict
emotionidx = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def Detect_img():
    if 'check' not in st.session_state:
        st.session_state.check = False
    if 'imag' not in st.session_state:
        st.session_state.imag = None
    if 'cnt' not in st.session_state:
        st.session_state.cnt = 0
    if 'emo' not in st.session_state:
        st.session_state.emo = []    
    if 'imgSave' not in st.session_state:
        st.session_state.imgSave = []
    
    st.title("Upload image")

    #Create path weight
    relative_path = r"Model\weight_face_emotion_1.pt"
    defaul_dir = os.path.dirname(__file__)
    file_path_weight = os.path.join(defaul_dir, relative_path)

    #load model
    device = torch.device('cpu')
    net = Deep_Emotion()
    net.load_state_dict(torch.load(file_path_weight, map_location=device))
    net.to(device)

    if st.session_state.check == False:
        # Tạo một khung tải lên tệp
        uploaded_file = st.file_uploader("JPG hoặc PNG", type=["jpg", "png"],)
    
        if uploaded_file is not None :
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error("Lỗi: Hình ảnh không hợp lệ. Vui lòng tải lên hình ảnh JPG hoặc PNG.")

            #show image
            st.write("Image")
            st.image(uploaded_file)

            opencv_image = np.array(image)
                
            #face detection
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
                sv_img = cv2.resize(graytemp, (500, 500))
                st.session_state.imgSave.append(sv_img)
                
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

                st.session_state.emo.append(int(torch.argmax(pred)))

                cv2.putText(opencv_image, status[int(torch.argmax(pred))], (x - 10, y - 20), font, 2, (0, 255, 0), 2, cv2.LINE_4)
            st.session_state.imag = opencv_image
            
        

    # Build estimation feature
    if st.session_state.imag is not None:
        global TrueIdx,FalseIdx
        st.write("Emotions of the objects in the image are predicted") 
        st.image(st.session_state.imag)  
        etm = st.button("Evaluate")
        

        if etm: 
            global file_path_trainTrue,file_path_trainFalse
            st.session_state.check=True

            #create serial image
            df = pd.read_csv(file_path_trainTrue)
            TrueIdx = df['rank'].max()
            if(str(TrueIdx).isdigit()): TrueIdx +=1
            else: TrueIdx = 1
            
            df = pd.read_csv(file_path_trainFalse)
            FalseIdx = df['rank'].max()
            if(str(FalseIdx).isdigit()): FalseIdx +=1
            else: FalseIdx =1
            
        if st.session_state.check:
            st.write("Is this really your emotion?")
            l,m,r = st.columns(3)
            # button position
            with m:
                nxt = st.button("Next")

            if nxt or st.session_state.check:
                if(st.session_state.cnt<len(st.session_state.emo)):
                    st.write(emotionidx[st.session_state.emo[st.session_state.cnt]])
                    st.image(st.session_state.imgSave[st.session_state.cnt])
                    st.session_state.cnt+=1
                    
                    with l:
                        yes = st.button("Yes")
                    with r:
                        no = st.button("No")

                    if yes:
                        #Save data to trainTrue.csv
                        if not os.path.exists(file_path_trainTrue):
                            with open(file_path_trainTrue, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(['rank','emotion', 'pixels'])
                        else:
                            with open(file_path_trainTrue, 'a') as f:
                                writer = csv.writer(f)
                                imgD = st.session_state.imgSave[st.session_state.cnt]
                                imgD2 = cv2.resize(imgD,(48,48))
                                flattened = imgD2.flatten()
                                pixel_values = ' '.join(map(str, flattened))
                                writer.writerow([TrueIdx,st.session_state.emo[st.session_state.cnt], pixel_values])
                                
                    if no:
                        #Save data to trainFalse.csv
                        if not os.path.exists(file_path_trainFalse):
                            with open(file_path_trainFalse, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(['rank','emotion', 'pixels'])
                        else:
                            with open(file_path_trainFalse, 'a') as f:
                                writer = csv.writer(f)
                                imgD = st.session_state.imgSave[st.session_state.cnt]
                                imgD2 = cv2.resize(imgD,(48,48))
                                flattened = imgD2.flatten()
                                pixel_values = ' '.join(map(str, flattened))
                                writer.writerow([FalseIdx,st.session_state.emo[st.session_state.cnt], pixel_values])

                else:
                    st.write("Please upload new image!!")
                    st.session_state.check = False
                    st.session_state.emo = []
                    st.session_state.imgSave = []
                    st.session_state.cnt = 0

                

                    

            
            
