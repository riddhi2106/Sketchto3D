from sklearn.datasets import load_digits

def load_image(index=0):
    digits = load_digits()
    return digits.images[index]