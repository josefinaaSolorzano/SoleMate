from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import os

from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf

# Custom DepthwiseConv2D that ignores the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the groups argument
        super().__init__(*args, **kwargs)

# Custom object scope
custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}

# Load the model with the custom object scope
model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def classify_waste(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

st.set_page_config(layout='wide')

st.title('_SoleMate_')
st.subheader('_Encontrá tu par perfecto_', divider='red')

video_file = open('NikeComercial.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

input_img = st.file_uploader("_Ingresá la foto del modelo que buscas y conocé más con un solo click_", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            if label == "0 Air Forces":
                st.success("The image is classified as Air Forces.")                
            elif label == "1 Air Jordans":
                st.success("The image is classified as Air Jordans.")
            elif label == "2 Air Maxes":
                st.success("The image is classified as Air Maxes.")
            elif label == "3 Cleats":
                st.success("The image is classified as Cleats.")
            elif label == "4 Dunks":
                st.success("The image is classified as Dunks.")
            else:
                st.error("The image is not classified as any relevant class.")

st.title('_Ya tenes tu par?_')
st.subheader('_Hacé click en los modelos para encontrar tu SoleMate_', divider='red')


col1, col2, col3, col4 = st.columns(4)

with col1:
   st.image("airfORces.jpg")
   if st.button("Air Forces", type="primary"):
    st.write('<a href="https://www.nike.com.ar/air%201%20force?_q=air%201%20force&map=ft" target="_blank">Haz clic aquí para ir a la página</a>', unsafe_allow_html=True)
   

with col2:
   st.image("airMaxestest.jpg")
   if st.button("Air Maxes", type="primary"):
    st.write('<a href="https://www.nike.com.ar/air%20max?_q=air%20max&map=ft" target="_blank">Haz clic aquí para ir a la página</a>', unsafe_allow_html=True)

with col3:
   st.image("jordan1.jpg")
   if st.button("Jordans", type="primary"):
    st.write('<a href="https://www.nike.com.ar/jordan?_q=jordan&map=ft&page=2" target="_blank">Haz clic aquí para ir a la página</a>', unsafe_allow_html=True)

with col4:
   st.image("dunkstest.jpg")
   if st.button("Dunks", type="primary"):
    st.write('<a href="https://www.nike.com.ar/dunks?_q=dunks&map=ft&page=2" target="_blank">Haz clic aquí para ir a la página</a>', unsafe_allow_html=True)
   
