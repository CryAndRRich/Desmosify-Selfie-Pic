import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convolve(image, kernel):
    r, c = image.shape
    ker_r, ker_c = kernel.shape

    padded = np.pad(image, (ker_r // 2, ker_c // 2), mode="constant")
    conv = np.empty_like(image, dtype=np.float32)
    if r * c < 2 ** 20:  # Fast convolution for small images
        stacked = np.array(
            [
                np.ravel(padded[i : i + ker_r, j : j + ker_c])
                for i in range(r)
                for j in range(c)
            ],
            dtype=np.float32,
        )

        conv = (stacked @ np.ravel(kernel)).reshape(r, c)
    else:  # Memory efficient convolution for large images
        for i in range(r - ker_r // 2):
            for j in range(c - ker_c // 2):
                conv[i][j] = np.sum(padded[i:i + ker_r, j:j + ker_c] * kernel)

    return conv

def sobel_edge_detection(image):
    gray_image = rgb2gray(image)
    
    # Horizontal and Vertical Sobel kernels
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32)
    
    # Normalize the vectors
    sobel_x = convolve(gray_image, Kx) / 8.0
    sobel_y = convolve(gray_image, Ky) / 8.0

    # Calculate the gradient magnitude
    grad_mag = (sobel_x ** 2 + sobel_y ** 2) ** 0.5

    sobel_edges = (grad_mag / np.max(grad_mag)) * 255
    return sobel_edges

