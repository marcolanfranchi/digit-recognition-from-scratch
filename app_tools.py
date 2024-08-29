from PIL import Image, ImageOps
import numpy as np

def zoom_on_digit(image, padding=10):
    """
    Zooms in on the drawn digit by cropping the image to the digit's bounding box.

    Parameters:
    - image: PIL.Image object, the input image containing the digit
    - padding: int, optional, the padding around the digit (default: 5 pixels)

    Returns:
    - zoomed_image: PIL.Image object, the zoomed-in image
    """
    # Convert image to numpy array
    img_array = np.array(image)

    # Find the non-black (non-zero) pixel positions
    non_black_pixels = np.argwhere(img_array > 0)

    if non_black_pixels.size == 0:
        # If there are no non-black pixels, return the original image
        return image

    # Get the bounding box of the non-black pixels
    top_left = non_black_pixels.min(axis=0)
    bottom_right = non_black_pixels.max(axis=0)

    # Add some padding
    top_left = np.maximum(top_left - padding, 0)
    bottom_right = np.minimum(bottom_right + padding, np.array(image.size) - 1)

    # Zoom the image to the bounding box
    cropped_image = image.crop((*top_left, *bottom_right))

    # Resize the cropped image to 28x28 (standard MNIST size)
    zoomed_image = cropped_image.resize((28, 28))

    return zoomed_image
