# libraries
import streamlit as st
from model import knn
from load_data import bytes_to_int
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from app_tools import zoom_on_digit
import pandas as pd
import datetime as dt
import plotly.figure_factory as ff

# ---------------------------- Title, Name, and Short Description ----------------------------------
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.title('Handwritten Digit Recognition with KNN')
st.markdown('''
            :grey[August 2024]
            ''')
st.write("")
st.write("")

st.markdown("This project explores handwritten digit recognition using the MNIST dataset and a K-nearest neighbors (KNN) \
         algorithm implemented from scratch in Python. In the `About` tab, you can learn about the dataset and algorithm implemented from scratch. \
         Then in the `Test Model` tab, you can test the model yourself by drawing a digit, running the knn algorithm, and seeing if the algorithm \
         correctly classifies your digit.")
# --------------------------------------------------------------------------------------------------


# initialize tabs
tab1, tab2 = st.tabs(['About', 'Test Model'])

with tab1: # ---------------------------------------------------------------------------------------
# Section 1: MNIST Dataset Overview
    st.header('Overview of the MNIST Dataset')
    st.write("The MNIST dataset contains 70,000 images of handwritten digits (0-9), where each image is represented as a 28x28 grid of \
            pixel values. Each pixel is stored as a  value ranging from 0 (black) to 255 (white), indicating the \
            grayscale colour at each point.")
    # Uncomment the line below to add an example image from the MNIST dataset
    st.image('images/mnist.png', caption='10 Samples from MNIST dataset', width=600)

    # streamlit text with hyperlink
    st.markdown("For more information on the MNIST dataset, go to the [MNIST website](http://yann.lecun.com/exdb/mnist/).")


    st.write("In order to feed these images into a machine learning model, each of the image's 2D grid of pixels were converted \
            into a 1D array. This flattening process transforms the 28x28 matrix into a single array of length 784. Here's a \
            snippet of the code:")

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

    st.write("After flattening the image's pixel matrix to an array, the data is in a format where we can calculate the distance \
            between the test image's and all of the training images from the MNIST dataset within the KNN algorithm.")

    # Section 2: KNN Algorithm on Flattened Images
    st.header('How the K-Nearest Neighbors Algorithm Works on MNIST Data')
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
        Compute distances between test sample and all training samples
        """
        return [dist(train_sample, test_sample) for train_sample in X_train]
    ''', language='python')

    st.write("The distance function computes the Euclidean distance between two vectors (here, the flattened pixel arrays). \
            The closer the distance, the more similar the images are considered to be. The get_training_distances function \
            calculates the distance between the provided test image and all training images.")

    # Walk through the main KNN function with more detail
    st.write("The core of the KNN algorithm is comparing the test image to all training images and identifying the \
            'k' nearest neighbors, i.e., the training images with the smallest distance to the test image. Here's a \
            snippet of how this works using the functions from above:")

    st.code('''
    def knn(X_train, y_train, X_test, k=3):
        """
        K-nearest neighbors algorithm, returns predictions for X_test
        """    
        predictions = []
        # Loop over each test sample
        for test_sample in X_test:
            # Calculate distances from the test sample to all training samples
            distances = get_training_distances(X_train, test_sample)
            # Sort the training samples by their distance to the test sample and get the indices of the k closest neighbors
            candidates = [y_train[i] for i in sorted(range(len(distances)), key=lambda i: distances[i])[:k]]
            # Find the most common label among the k nearest neighbors and use it as the prediction for this test sample
            predictions.append(max(set(candidates), key=candidates.count))
        return predictions # return the list of predictions
    ''', language='python')


    st.write("The algorithm calculates the distance from the test image to each training image, finds the 'k' nearest \
            images, and assigns the most frequent label among these neighbors as the predicted class for the test image.")

    # Section 3: Putting It All Together
    st.header('Summary of KNN for Handwritten Digit Recognition')
    st.write("In summary, the workflow for recognizing a handwritten digit with KNN using the the MNIST dataset involves:")
    st.write("""
    1. **Reading and Flattening the Images**: Convert the 28x28 pixel matrices into 1D arrays.
    2. **Calculating Distances**: Compute the Euclidean distance between the test image and all training images.
    3. **Identifying Neighbors**: Sort the distances to find the 'k' nearest neighbors.
    4. **Voting**: Take a majority vote among the neighbors to determine the predicted digit.
    """)

with tab2: # ---------------------------------------------------------------------------------------

    # Interactive Testing Part
    st.header('Test the Model')

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([5, 1, 5])

    with col1:

        img_col1, img_col2 = st.columns(2)

        with img_col1:

            st.write("Draw a digit in the canvas below:")
            canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=16,
            stroke_color="#fff",
            background_color="#000",
            update_streamlit=True,
            height=224,
            width=224,
            drawing_mode="freedraw",
            key="canvas_1"
            )


        with img_col2:
            st.write("Your drawing:")
            # Do something interesting with the image data and paths
            image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            image = image.convert('L')
            # image = zoom_on_digit(image)
            image = image.resize((28, 28))
            st.image(image, use_column_width=True, clamp=True)
        
        st.subheader("KNN Model Parameters")
        
        # Allow the user to change the value of k
        k_value = int(st.text_input("Edit the value of k:", value=3))
        

        # Display the code with an editable input for k
        st.code('''
        knn(X_test=[image], k={})
        '''.format(k_value), language='python')

        test = st.button("Test the model")
        if test:
            st.write("Classifying the uploaded image...")
            image = np.array(image).flatten()
            # convert image to bytes
            image = [bytes([pixel]) for pixel in image]

            # Run the KNN model on the uploaded image
            prediction = knn(X_test = [image], k=k_value)
            st.write("Recognized digit:", bytes_to_int(prediction[0]))

    # --------------------------------------------------------------------------------------------------
    with col2:
        st.write("") # empty column for blank space


    # --------------------------------------------------------------------------------------------------
    with col3:
        st.write("KNN Visualization:")
        # placeholder stuff below
        # Add histogram data
        x1 = np.random.randn(200) - 2
        x2 = np.random.randn(200)
        x3 = np.random.randn(200) + 2

        # Group data together
        hist_data = [x1, x2, x3]

        group_labels = ['Group 1', 'Group 2', 'Group 3']

        # Create distplot with custom bin_size
        fig = ff.create_distplot(
                hist_data, group_labels, bin_size=[.1, .25, .5])

        # Plot!
        st.plotly_chart(fig, use_container_width=True)
