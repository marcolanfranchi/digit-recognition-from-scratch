# Libraries ---------------------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
# My functions
from model.model import knn
from model.load_data import bytes_to_int
from ui.app_tools import zoom_on_digit
from ui.explanations import overview_dropdown, knn_dropdown, summary_dropdown
from model.load_data import read_images

# Data ---------------------------------------------------
DATA_DIR = 'data/'
TRAIN_DATA_FILE = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATA_DIR + 'train-labels-idx1-ubyte'
TEST_DATA_FILE = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABEL_FILE = DATA_DIR + 't10k-labels-idx1-ubyte'
X_train_2D = np.load('data/X_train_2D.npy')
y_train = np.load('data/y_train.npy')

# Page Configurations -----------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# remove streamlit's main menu and footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


# Title, Name + Date, and Short Description -----------------------------------------------------------
st.title('Handwritten Digit Recognition with KNN')
st.markdown('''
            :grey[`ML. Aug 2024`]
            ''')
st.write("")

st.markdown("This project explores handwritten digit recognition using the MNIST \
            dataset and the K-nearest neighbours (KNN) algorithm implemented from scratch \
            in Python. In the drop-down sections below, \
            you can learn about the dataset and the algorithm. \
            In the [`Test Model`](#test-the-model) section, you can test \
            the model yourself by drawing a digit, running the KNN algorithm, \
            and seeing if it correctly recognizes your digit.")


# Explanations Section ------------------------------------------------------------------
overview_dropdown()
knn_dropdown()
summary_dropdown()


# Interactive Testing Section ------------------------------------------------------------------
st.write("---")
st.header('Test the Model')
st.write("")

col1, col2, col3, col4, col5 = st.columns([1, 0.1, 1, 0.1, 2])

with col1: # --------------------------------------------------------------------------------------------------
    st.write("Draw a digit (0-9) below:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  
        stroke_width=17,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=224,
        width=224,
        drawing_mode="freedraw",
        key="canvas"
    )

    st.write("---")
    st.write("Your drawing:")
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        image = image.convert('L')
        # image = zoom_on_digit(image)
        image = image.resize((28, 28))
        st.image(image, width=224)
    
with col2: # --------------------------------------------------------------------------------------------------
    pass

with col3: # --------------------------------------------------------------------------------------------------
    st.write("Specify KNN parameters:")
    
    # Allow the user to change the value of k
    k_value = int(st.slider("Edit the value of k:", value=5, min_value=1, max_value=10, step=1))

    # Display the code with an editable input for k
    # st.write("KNN Algorithm Call:")
    st.code(f'''
    knn(X_test=[drawing], k={k_value})
    ''', language='python')

    st.write("")
    test = st.button("Test the model", help="Click to classify the digit in your drawing")
    st.write("")
    st.write("")
    st.write("---")

    st.write("Results:")

    if test:
        with st.spinner("Classifying your drawing ..."):
            image = np.array(image).flatten()
            image = [bytes([pixel]) for pixel in image]

            # Run the KNN model on the uploaded image
            prediction, neighbour_indices = knn(X_test=[image], k=k_value)
            recognized_digit = bytes_to_int(prediction[0])

            # Display the recognized digit 
            st.metric(label="Recognized Digit:", 
                    value=recognized_digit, 
                    delta="", 
                    help="The digit recognized in your drawing by the KNN model")
        

with col4: # --------------------------------------------------------------------------------------------------
    pass

with col5: # --------------------------------------------------------------------------------------------------

    st.write("K-Nearest Neighbours (KNN) images:")
    if test:
        # Create columns based on k_value
        columns = st.columns(k_value, vertical_alignment='center')

        # Loop through the columns and add images to each column
        for i, idx in enumerate(neighbour_indices[0]):
            # Load and display the image in the corresponding column
            image = Image.fromarray(read_images(TRAIN_DATA_FILE)[idx], 'L')
            with columns[i]:  # Place the image in the i-th column
                st.image(image, width=100)  # Adjust width as needed
    else:
        st.markdown(":grey[Draw a digit and click the 'Test the model' button to see the nearest neighbors.]")
    st.write("---")
    # st.write("t-SNE Visualization of MNIST Data:")

    # create a scatter plot of the 2D t-SNE data
    fig = go.Figure()

    # Add scatter plot for all training data
    fig.add_trace(go.Scatter(
        x=X_train_2D[:, 0],
        y=X_train_2D[:, 1],
        mode='markers',
        marker=dict(color=y_train, colorscale='Turbo', showscale=False),
        text=[f"Digit: {digit}" for digit in y_train],  # Tooltip to show the digit on hover
        hoverinfo='text',
        name='Training Data'
    ))

    # add legend
    fig.update_layout(
        # showlegend=True,
        legend_title_text='Digit',
        title="t-SNE Visualization of MNIST Data",
        title_x=0.25,
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2"
    )

    # plot
    st.plotly_chart(fig, use_container_width=True, )
