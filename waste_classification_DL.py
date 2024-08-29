import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

hide_streamlit_style = """
    <style>
    #GithubIcon {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {overflow: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def load_and_preprocess_image(image, img_size):
    # Convert the image to an array
    image = np.array(image)
    
    # Resize the image
    resized_img = cv2.resize(image, img_size)
    
    # Normalize the image
    normalized_img = resized_img / 255.0
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(normalized_img, axis=0)
    
    return img_array

def predict_image_class(interpreter, image, img_size):
    img_array = load_and_preprocess_image(image, img_size)
    
    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Interpret the result
    if output_data[0] > 0.5:
        result = 'Non-Biodegradable'
    else:
        result = 'Biodegradable'
    
    return result

# Streamlit App
st.subheader("Waste Classification App",divider=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Image size for prediction
img_size = (100,100)  # Example size, modify as per your model input

# Load the TFLite model
tflite_model_path = "vgg16model.tflite"  # Path to your TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Placeholder for the button
    if st.button('Submit'):
        # Make prediction
        result = predict_image_class(interpreter, image, img_size)
        
        # Display the prediction result
        st.subheader(f'The image is classified as: {result}')
