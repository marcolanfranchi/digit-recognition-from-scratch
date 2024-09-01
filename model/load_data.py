# Libraries
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Data ---------------------------------------------------
DATA_DIR = 'data/'
TRAIN_DATA_FILE = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATA_DIR + 'train-labels-idx1-ubyte'
TEST_DATA_FILE = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABEL_FILE = DATA_DIR + 't10k-labels-idx1-ubyte'


def bytes_to_int(bytes):
    """
    Convert byte value to integer
    """
    return int.from_bytes(bytes, byteorder='big')


def read_images(filename):
    """
    Read images from file and return as list
    """
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # pass by the magic number 
        n_images = bytes_to_int(f.read(4))
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))

        # read the pixel values
        images = np.frombuffer(f.read(n_images * n_rows * n_cols), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)

    return images


def read_labels(filename):
    """
    Read labels from file and return as list
    """
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # pass by the magic number
        n_labels = bytes_to_int(f.read(4))
        
        # read the label data
        labels = np.frombuffer(f.read(n_labels), dtype=np.uint8)

    return labels


def extract_features(X):
    """
    Flatten 2D list of pixel values to 1D list
    """
    n_images, n_rows, n_cols = X.shape
    # flatten each 2D image to 1D
    features = X.reshape(n_images, n_rows * n_cols)
    
    return features


def generate_tsne_data():
    """
    Generate the t-SNE data for the MNIST dataset and save it to a file.
    """
    # Load your data
    X_train_images = read_images(TRAIN_DATA_FILE)
    y_train = read_labels(TRAIN_LABEL_FILE)
    X_train_features = extract_features(X_train_images)

    # # Apply PCA to reduce dimensions
    # pca = PCA(n_components=50)
    # X_train_pca = pca.fit_transform(X_train_features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_train_2D = tsne.fit_transform(X_train_features)

    # Save the t-SNE results and labels
    np.save('data/X_train_2D.npy', X_train_2D)
    np.save('data/y_train.npy', y_train)

def main():
    generate_tsne_data()

if __name__ == "__main__":
    main()
