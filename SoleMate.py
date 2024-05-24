from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
#from openai import OpenAI



def classify_fruit(img):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
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
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    return class_name, confidence_score



#CATA def classify_waste(img):
    # Disable scientific notation for clarity
    # np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model
    # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    # image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
   #size = (224, 224)
   # image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
#    image_array = np.asarray(image)

    # Normalize the image
    #normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    #data[0] = normalized_image_array

    # Predicts the model
    #prediction = model.predict(data)
    #index = np.argmax(prediction)
    #class_name = class_names[index].strip()
    #confidence_score = prediction[0][index]

    #return class_name, confidence_score

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
            label, confidence_score = classify_fruit(image_file)
            if label == "0 Air Forces":
                st.success("Tu par perfecto es Air Forces.")                
            elif label == "1 Air Jordans":
                st.success("Tu par perfecto es Air Jordans.")
            elif label == "2 Air Maxes":
                st.success("Tu par perfecto es Air Maxes.")
            elif label == "3 Cleats":
                st.success("Tu par perfecto es Cleats.")
            elif label == "4 Dunks":
                st.success("Tu par perfecto es Dunks.")
            else:
                st.error("No encontramos ningún match para vos 😢 Cargá otra foto para que encontremos tu par ideal! .")

