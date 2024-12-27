import numpy as np 

# Converts an RGB image into grayscale
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

# Create Gaussian kernel
def gaussian_kernel(sigma):
    size = np.ceil(3 * sigma).astype(int)

    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * (sigma ** 2))
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))) * normal

    return kernel

# Edge detection using zero crossing algorithm 
def zero_crossing(image, threshold):
    h, w = image.shape

    edges = np.zeros_like(image, dtype=np.uint8)
    # For each pixel, calculate difference between two adjusted pixels
    for i in range(h):
        for j in range(w):
            if i > 0 and i < h - 1:
                left = image[i - 1, j]
                right = image[i + 1, j]
                if left * right < 0 and np.abs(left - right) > threshold:
                    edges[i, j] = 255 # If difference is above a threshold, it is marked as an edge pixel
            if j > 0 and j < w - 1:
                up = image[i, j + 1]
                down = image[i, j - 1]
                if up * down < 0 and np.abs(up - down) > threshold:
                    edges[i, j] = 255
            if (i > 0 and i < h - 1) and (j > 0 and j < w - 1):
                UL = image[i - 1, j - 1]
                UR = image[i + 1, j - 1]
                BL = image[i - 1, j + 1]
                BR = image[i + 1, j + 1]
                if (UL * BR < 0 and np.abs(UL - BR) > threshold) or \
                    (BL * UR < 0 and np.abs(BL - UR) > threshold):

                    edges[i, j] = 255
    
    return edges

def marr_hildreth_edge_detection(image, sigma=1.4, threshold=15):
    gray_image = rgb2gray(image)

    # Filtered with a Laplacian of Gaussian (LoG) kernel
    laplacian_kernel = np.array([[1, 1, 1], 
                                 [1, -8, 1], 
                                 [1, 1, 1]], dtype=np.float32)
    gauss_ker = gaussian_kernel(sigma)
    image_blurred = convolve(gray_image, gauss_ker)
    image_laplacian = convolve(image_blurred, laplacian_kernel)

    # Apply zero crossing algorithm 
    marr_hildreth_edges = zero_crossing(image_laplacian, threshold)

    return marr_hildreth_edges

