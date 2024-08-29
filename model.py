from load_data import bytes_to_int, read_images, read_labels, extract_features

# --------------------------------------------------------
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
    return sum([(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)]) ** 0.5


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
    # X_test = read_images(TEST_DATA_FILE)[:5]

    X_train = extract_features(X_train)
    # X_test = extract_features(X_test)


    y_pred = []
    for sample in X_test:
        distances = get_training_distances(X_train, sample)
        sorted_indices = [
            pair[0]
            for pair in sorted(
                enumerate(distances), 
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_indices[:k]
        ]
        top_candidate = max(candidates, key=candidates.count)
        y_pred.append(top_candidate)
    return y_pred


def main():
    # X_train = read_images(TRAIN_DATA_FILE)
    # y_train = read_labels(TRAIN_LABEL_FILE)
    X_test = read_images(TEST_DATA_FILE)[:5]
    X_test = extract_features(X_test)
    # y_test = read_labels(TEST_LABEL_FILE)[:5]

    # X_train = extract_features(X_train)
    # X_test = extract_features(X_test)

    # y_test = read_labels(TEST_LABEL_FILE)[:5]

    # predictions = knn(TRAIN_DATA_FILE, TRAIN_LABEL_FILE, TEST_LABEL_FILE, 3)
    # accuracy = sum([1 for y_pred, y_true in zip(predictions, y_test) if y_pred == y_true]) / len(y_test)
    # print('Accuracy:', accuracy)
    # for y_pred, y_true in zip(predictions, y_test):
    #     print('Predicted:', bytes_to_int(y_pred), 'True:', bytes_to_int(y_true))
    # save X_test to file
    print(X_test[0])


if __name__ == '__main__':
    main()