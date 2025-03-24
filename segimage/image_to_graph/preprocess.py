import cv2


def preprocess_image(image_path):
    """Load and convert the image to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

