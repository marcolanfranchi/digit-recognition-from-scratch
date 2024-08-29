import streamlit as st
import numpy as np
from PIL import Image
from model import knn

# init app
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.title('Handwritten Digit Recognition')

st.write('This app recognizes handwritten digits using a K-nearest neighbors algorithm written from scratch in Python.')

st.write('Upload an image of a handwritten digit to get started!')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write("Classifying...")

    # Read image
    image = Image.open(uploaded_file)
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)

    # Flatten image
    image = image.flatten()