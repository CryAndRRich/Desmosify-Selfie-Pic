import numpy as np

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale

    Parameters:
        rgb: The input image in RGB format as a 2D array of shape (height, width, 3)

    --------------------------------------------------
    Returns:
        np.ndarray: A 2D grayscale image of the same size as the input image
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convolve(image: np.ndarray, 
             kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a given kernel

    Parameters:
        image: The input image to be convolved, a 2D numpy array
        kernel: The kernel to be used for the convolution, a 2D numpy array

    --------------------------------------------------
    Returns:
        conv: The convolved image as a 2D numpy array of the same size as the input image
    """
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

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Perform edge detection using the Sobel operator

    Parameters:
        image: The input image in RGB format

    --------------------------------------------------
    Returns:
        sobel_edges: A binary image with edges marked as 255 and non-edges marked as 0
    """
    gray_image = rgb2gray(image)
    
    # Horizontal and Vertical sobel kernels
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

    # Normalize and scale to the range 0-255
    sobel_edges = (grad_mag / np.max(grad_mag)) * 255

    return sobel_edges
