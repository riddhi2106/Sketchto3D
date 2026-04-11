import cv2

def detect_edges(image):
    # Apply slight blur to improve edge detection
    blurred = cv2.GaussianBlur(image, (5,5), 0)

    # Use LOWER thresholds (important!)
    edges = cv2.Canny(blurred, 30, 100)

    return edges