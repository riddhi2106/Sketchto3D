import cv2
import numpy as np

def preprocess_image(image):
    # Convert to 0–255 grayscale
    image = (image / image.max()) * 255
    image = image.astype(np.uint8)

    # Resize for better edge detection
    image = cv2.resize(image, (128, 128))

    return image