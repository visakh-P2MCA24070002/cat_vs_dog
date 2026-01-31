import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_dog_model.h5')
    return model

model = load_model()

# Define class names
class_names = ['Cat', 'Dog']

st.title('Cat vs. Dog Image Classifier')
st.write('Upload an image of a cat or a dog, and the model will predict what it is!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = img.resize((150, 150)) # Resize to target size used in training
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100

    st.success(f'Prediction: {predicted_class} with {confidence:.2f}% confidence.')
