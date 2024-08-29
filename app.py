import streamlit as st
from model import knn
from load_data import bytes_to_int
from PIL import Image
import numpy as np

st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.title('Handwritten Digit Recognition with KNN')

st.write("This project explores handwritten digit recognition using the MNIST dataset and a K-nearest neighbors (KNN) \
         algorithm implemented from scratch in Python. You can learn about the dataset and algorithm below, \
         then test the model yourself by drawing a digit, running the knn algorithm, and seeing if the algorithm \
         arrives at the correct classification of your digit.")

# Section 1: MNIST Dataset Overview
st.header('Understanding the MNIST Dataset')
st.write("The MNIST dataset consists of 70,000 images of handwritten digits (0-9), where each image is a 28x28 pixel grid. \
         This grid represents the intensity of the pixels, ranging from 0 (black) to 255 (white).")
# Uncomment the line below to add an example image from the MNIST dataset
# st.image('path_to_image', caption='Sample MNIST Digit', use_column_width=True)

st.write("In order to feed these images into a machine learning model, the 2D grid of pixels were converted into a 1D array. \
         This flattening process transforms the 28x28 matrix into a single array of length 784, where each element \
         corresponds to a pixel intensity.")

st.code('''
def flatten_list(list):
    """
    Flatten list of lists to 1D list
    """
    return [item for sublist in list for item in sublist]
    
# Example:
# Original Image (2D Array)
# [[12, 50, 100],
#  [34, 89, 123]]

# Flattened Image (1D Array)
# [12, 50, 100, 34, 89, 123]
''', language='python')

st.write("After flattening the image, the data is in a format where we can calculate the distance \
        between the flattened array of the test image and all training images in the KNN algorithm.")

# Section 2: KNN Algorithm on Flattened Images
st.header('How the K-Nearest Neighbors (KNN) Algorithm Works on MNIST Data')
st.write("Once the images pixels are flattened to a 1D array, KNN can be used to classify the \
         handwritten digits. For each test image, the algorithm computes the distance between the \
         flattened pixel array of the test image and the flattened arrays of all training images.")

# Explain the distance function
st.code('''
def dist(x, y):
    """
    Compute Euclidean distance between two vectors
    """
    return sum([(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)]) ** 0.5


def get_training_distances(X_train, test_sample):
    """
    Compute distance between test sample and all training samples
    """
    return [dist(train_sample, test_sample) for train_sample in X_train]
''', language='python')

st.write("The distance function computes the Euclidean distance between two vectors (here, the flattened pixel arrays). \
         The closer the distance, the more similar the images are considered to be.")

# Walk through the main KNN function with more detail
st.write("The core of the KNN algorithm is comparing the test image to all training images and identifying the \
         'k' nearest neighbors, i.e., the training images with the smallest distance to the test image. Here's a \
         snippet of how this works:")

st.code('''
def knn(X_train, y_train, X_test, k=3):
    predictions = []
    for test_sample in X_test:
        distances = [dist(train_sample, test_sample) for train_sample in X_train]
        nearest_labels = [y_train[i] for i in sorted(range(len(distances)), key=lambda i: distances[i])[:k]]
        predictions.append(max(set(nearest_labels), key=nearest_labels.count))
    return predictions
''', language='python')

st.write("The algorithm calculates the distance from the test image to each training image, finds the 'k' nearest \
         images, and assigns the most frequent label among these neighbors as the predicted class for the test image.")

# Section 3: Putting It All Together
st.header('Putting It All Together: KNN for Digit Recognition')
st.write("In summary, the workflow for recognizing a handwritten digit with KNN on the MNIST dataset involves:")
st.write("""
1. **Reading and Flattening the Images**: Convert the 28x28 pixel matrices into 1D arrays.
2. **Calculating Distances**: Compute the Euclidean distance between the test image and all training images.
3. **Identifying Neighbors**: Sort the distances to find the 'k' nearest neighbors.
4. **Voting**: Take a majority vote among the neighbors to determine the predicted digit.
""")

# Interactive Testing Part
st.header('Test the Model')
st.write("You can now upload your own handwritten digit image to see how well the model performs.")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False)
    st.write("Classifying the uploaded image...")
    
    # Read and preprocess the uploaded image
    image = Image.open(uploaded_file)
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).flatten()
    # convert image to bytes
    image = [bytes([pixel]) for pixel in image]
    # st.write(image)

    # Placeholder for running the KNN model on the uploaded image
    prediction = knn(X_test = [image], k=3)
    st.write("Predicted digit:", bytes_to_int(prediction[0]))
