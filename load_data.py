

def bytes_to_int(bytes):
    """
    Convert bytes to integer
    """
    return int.from_bytes(bytes, byteorder='big')


def read_images(filename):
    """
    Read images from file
    """
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
    """
    Read labels from file
    """
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        for label_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels


def flatten_list(list):
    """
    Flatten list of lists to 1D list
    """
    return [item for sublist in list for item in sublist]


def extract_features(X):
    """
    Extract features from images
    """
    return [flatten_list(x) for x in X]
