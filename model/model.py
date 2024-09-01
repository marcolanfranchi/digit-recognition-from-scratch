# Libraries (numpy only for optimization)
import numpy as np
from model.load_data import bytes_to_int, bytes_to_int, read_images, read_labels, extract_features

# data ---------------------------------------------------
DATA_DIR = 'data/'
TRAIN_DATA_FILE = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATA_DIR + 'train-labels-idx1-ubyte'
TEST_DATA_FILE = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABEL_FILE = DATA_DIR + 't10k-labels-idx1-ubyte'
# --------------------------------------------------------

def dist(x, y):
    """
    Compute Euclidean distance between two vectors
    """
    x = np.array([bytes_to_int(pixel) for pixel in x])
    y = np.array([bytes_to_int(pixel) for pixel in y])
    return np.linalg.norm(x - y)

def get_training_distances(X_train, test_sample):
    """
    Compute distance between test sample and all training samples
    """
    return [dist(train_sample, test_sample) for train_sample in X_train]


def knn(X_train=TRAIN_DATA_FILE, y_train=TRAIN_LABEL_FILE, X_test=TEST_DATA_FILE, k=3):
    """
    K-nearest neighbors algorithm for classifying handwritten digits using the MNIST dataset
    """
    X_train = read_images(TRAIN_DATA_FILE)
    y_train = read_labels(TRAIN_LABEL_FILE)
    X_train = extract_features(X_train)

    # For testing
    # X_test = read_images(TEST_DATA_FILE)
    # X_test = extract_features(X_test)

    y_pred = []
    neighbor_indices = []
    for sample in X_test:
        distances = get_training_distances(X_train, sample)

        # get indices of the k smallest distances
        sorted_indices = np.argsort(distances)[:k]
        neighbor_indices.append(sorted_indices)

        # get their corresponding digit labels
        candidates = [y_train[idx] for idx in sorted_indices]
        
        # find most frequent label among the k nearest neighbors
        top_candidate = max(set(candidates), key=candidates.count)

        y_pred.append(top_candidate)

    return y_pred, neighbor_indices


def main():
    # X_train = read_images(TRAIN_DATA_FILE)
    # y_train = read_labels(TRAIN_LABEL_FILE)
    # X_test = read_images(TEST_DATA_FILE)[:5]
    # y_test = read_labels(TEST_LABEL_FILE)[:5]

    # X_train = extract_features(X_train)
    # X_test = extract_features(X_test)

    # predictions = knn(TRAIN_DATA_FILE, TRAIN_LABEL_FILE, TEST_LABEL_FILE, 3)
    # accuracy = sum([1 for y_pred, y_true in zip(predictions, y_test) if y_pred == y_true]) / len(y_test)
    # print('Accuracy:', accuracy)
    pass

if __name__ == '__main__':
    main()