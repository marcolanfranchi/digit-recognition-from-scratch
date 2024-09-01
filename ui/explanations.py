# Libraries ---------------------------------------------------
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Section 1: MNIST Dataset Overview  # ----------------------------------------------------------
def overview_dropdown():
    with st.expander("Overview of the MNIST Dataset", icon='🔢'):
        st.header('Overview of the MNIST Dataset')
        st.write("The MNIST dataset contains 70,000 images of handwritten digits (0-9), \
                where each image is represented as a 28x28 grid of \
                pixel values. Each pixel is stored as a byte value ranging from 0 (black) \
                to 255 (white), indicating the grayscale colour at each point.")
        # Uncomment the line below to add an example image from the MNIST dataset
        st.image('images/mnist.png', caption='10 Samples from MNIST dataset', width=600)

        # streamlit text with hyperlink
        st.markdown("For more information on the MNIST dataset, go to the \
                    [MNIST website](http://yann.lecun.com/exdb/mnist/).")


        st.write("In order to feed these images into a machine learning model, each of the \
                image's 2D grid of pixels were converted \
                into a 1D array. This flattening process transforms the 28x28 matrix into \
                a single array of length 784. Here's the \
                function used to flatten the pixel data for all images:")

        st.code('''
        def extract_features(X):
            """
            Flatten 2D list of pixel values to 1D list
            """
            n_images, n_rows, n_cols = X.shape
            # Flatten each 2D image to 1D
            features = X.reshape(n_images, n_rows * n_cols)
            
            return features
            
        # Example on a single image:
        # Original Image (2D Array)
        # [[12, 50, 100],
        #  [34, 89, 123]]

        # Flattened Image (1D Array)
        # [12, 50, 100, 34, 89, 123]
        ''', language='python')

        st.write("After flattening the image's pixel matrix to an array, the data is \
                in a format where we can calculate the distance \
                between the test image's and all of the training images from the MNIST \
                dataset inside the KNN algorithm.")

# Section 2: KNN Algorithm on Flattened Images ---------------------------------------------------------------------------------------
def knn_dropdown():
    with st.expander("How the K-Nearest Neighbours Algorithm Works on MNIST Data", icon='📈'): 
        st.header('How the K-Nearest Neighbours Algorithm Works on MNIST Data')
        st.write("Once the images pixels are flattened to a 1D array, KNN can be used to \
                classify the handwritten digits. For each test image, the algorithm computes \
                the distance between the flattened pixel array of the test image and \
                the flattened arrays of all training images.")

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

        st.write("The distance function computes the Euclidean distance between \
                two vectors (here, the flattened pixel arrays). \
                The closer the distance, the more similar the images are considered \
                to be. The get_training_distances function calculates the distance \
                between the provided test image and all training images.")

        # Walk through the main KNN function with more detail
        st.write("The core of the KNN algorithm is comparing the test image to all \
                training images and identifying the 'k' nearest neighbours, i.e., \
                the training images with the smallest distance to the test image. Here's a \
                snippet of how this works using the functions from above:")

        st.code('''
        def knn(X_train, y_train, X_test, k=3):
            """
            K-nearest neighbours algorithm, returns predictions for X_test
            """    
            predictions = []
            # Loop over each test sample
            for test_sample in X_test:
                # Calculate distances from the test sample to all training samples
                distances = get_training_distances(X_train, test_sample)
                # Sort the training samples by their distance to the test sample and get the indices of the k closest neighbours
                candidates = [y_train[i] for i in sorted(range(len(distances)), key=lambda i: distances[i])[:k]]
                # Find the most common label among the k nearest neighbours and use it as the prediction for this test sample
                predictions.append(max(set(candidates), key=candidates.count))
            return predictions # return the list of predictions
        ''', language='python')


        st.write("The algorithm calculates the distance from the test image to each \
                training image, finds the 'k' nearest images, and assigns the most \
                frequent label among these neighbours as the predicted class for the test image.")


# Section 3: Putting It All Together  ---------------------------------------------------------------------------------------
def summary_dropdown():
    with st.expander("Summary of KNN for Handwritten Digit Recognition", icon='📝'): 
        st.header('Summary of KNN for Handwritten Digit Recognition')
        st.write("In summary, the workflow for recognizing a handwritten digit with KNN using \
                the the MNIST dataset involves:")
        st.write("""
        1. **Reading and Flattening the Images**: Convert the 28x28 pixel matrices into 1D arrays.
        2. **Calculating Distances**: Compute the Euclidean distance between the test image and all training images.
        3. **Identifying Neighbours**: Sort the distances to find the 'k' nearest neighbours.
        4. **Voting**: Take a majority vote among the neighbours to determine the predicted digit.
        """)