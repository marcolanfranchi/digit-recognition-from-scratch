
DATA_DIR = 'data/'
TRAIN_DATA_FILE = DATA_DIR + 'train-images-idx3-ubyte'
TRAIN_LABEL_FILE = DATA_DIR + 'train-labels-idx1-ubyte'
TEST_DATA_FILE = DATA_DIR + 't10k-images-idx3-ubyte'
TEST_LABEL_FILE = DATA_DIR + 't10k-labels-idx1-ubyte'

def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='big')

def read_images(filename):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_cols):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels


def main():
    X_train = read_images(TRAIN_DATA_FILE)
    y_train = read_labels(TRAIN_LABEL_FILE)
    X_test = read_images(TEST_DATA_FILE)
    y_test = read_labels(TEST_LABEL_FILE)

    


if __name__ == '__main__':
    main()