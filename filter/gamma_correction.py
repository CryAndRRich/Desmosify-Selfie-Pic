import numpy as np

def gamma_correction(img, gamma = 1.0):
    img = np.power(img, gamma)
    max_val = np.max(img.ravel())
    img = img/max_val * 255
    img = img.astype(np.uint8)

    return img