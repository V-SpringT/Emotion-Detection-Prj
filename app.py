import streamlit as st
from Data_Colection import webcam_input
from Detect_by_camera import webcam_detect1
from Detect_by_image import Detect_img
from Emotion_Analysis import webcam_detect

st.title("Emotion Detection")
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
selected_emotion = emotions[0]
Features =  ["Home","Data Collection","Emotion Detection By Image","Emotion Detection By Real-time Video","Emotion Analysis"]
feature = st.sidebar.selectbox("Choose Feature",Features)
if feature == "Home":
    st.image(r"https://images.giaoducthoidai.vn/1200x630/Uploaded/2023/mfnms/2021-03-01/34-1.jpg",use_column_width=True)
if feature == "Data Collection":
    st.sidebar.header("Options")
    selected_emotion = st.sidebar.selectbox("Select Emotion",emotions)
    webcam_input(selected_emotion)
elif feature == "Emotion Detection By Image":
    Detect_img()
elif feature == "Emotion Detection By Real-time Video":
    webcam_detect1()
elif feature == "Emotion Analysis":
    webcam_detect()

