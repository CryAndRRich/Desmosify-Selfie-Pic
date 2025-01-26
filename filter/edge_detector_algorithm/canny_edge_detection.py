from typing import Tuple
import numpy as np 
from scipy import ndimage

# Using built-in function
#-----------------------------------------------------#
#def canny_edge_detection(img):                       |
#    canny_edges = cv2.Canny(img, 100, 150)           |
#    return canny_edges                               |
#-----------------------------------------------------#

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image into grayscale using the standard luminance coefficients

    Parameters:
        rgb: The input image in RGB format (3-channel color image)

    --------------------------------------------------
    Returns:
        np.ndarray: The resulting grayscale image
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def gaussian_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a Gaussian filter for noise reduction to the input image

    Parameters:
        image: The input image (grayscale)

    --------------------------------------------------
    Returns:
        image_blurred: The blurred image after applying the Gaussian filter
    """
    # Create Gaussian kernel
    size = 5
    sigma = 1.4
    
    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * (sigma ** 2))
    gaussian_kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))) * normal
    
    gray_image = rgb2gray(image)  # Convert image to grayscale
    
    image_blurred = ndimage.convolve(gray_image, gaussian_kernel)  # Apply Gaussian filter
    return image_blurred

def sobel_filters(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient magnitudes and directions using Sobel filters

    Parameters:
        image: The input image (grayscale)

    --------------------------------------------------
    Returns:
        grad_mag: The gradient magnitudes of the image
        theta: The gradient directions (angles) of the image
    """
    # Sobel kernels for edge detection
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32)
    
    # Compute gradients in x and y directions
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    
    # Compute gradient magnitudes and directions (angles)
    grad_mag = np.hypot(Ix, Iy)  # Magnitude of the gradient
    theta = np.arctan2(Iy, Ix)  # Angle of the gradient
    
    return (grad_mag, theta)

def non_maximum_suppress(grad_mag: np.ndarray, 
                         theta: np.ndarray) -> np.ndarray:
    """
    Perform non-maximum suppression to thin the edges

    Parameters:
        grad_mag: The gradient magnitude of the image
        theta: The gradient direction (angle) of the image

    --------------------------------------------------
    Returns:
        np.ndarray: The image after non-maximum suppression (thinned edges)
    """
    h, w = grad_mag.shape
    non_max_img = np.zeros((h, w), dtype=np.int32)
    
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180  # Normalize angles to [0, 180]

    # Loop through every pixel in the image for non-maximum suppression
    for i in range(1, h):
        for j in range(1, w):
            q = 255
            r = 255
            try:
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = grad_mag[i, j + 1]
                    r = grad_mag[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = grad_mag[i + 1, j - 1]
                    r = grad_mag[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = grad_mag[i + 1, j]
                    r = grad_mag[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = grad_mag[i - 1, j - 1]
                    r = grad_mag[i + 1, j + 1]
                
                # Non-maximum suppression step 
                if (grad_mag[i, j] >= q) and (grad_mag[i, j] >= r):
                    non_max_img[i, j] = grad_mag[i, j]
                else:
                    non_max_img[i, j] = 0
            except:
                pass

    return non_max_img

def double_thres(image: np.ndarray, 
                 grad_mag: np.ndarray, 
                 weak_pix: int, 
                 strong_pix: int, 
                 weak_ratio: float, 
                 strong_ratio: float) -> np.ndarray:
    """
    Apply double thresholding to the image to classify pixels as weak or strong edges

    Parameters:
        image: The input image (grayscale)
        grad_mag: The gradient magnitudes of the image
        weak_pix: The pixel value representing weak edges
        strong_pix: The pixel value representing strong edges
        weak_ratio: The threshold ratio for weak edges
        strong_ratio: The threshold ratio for strong edges

    --------------------------------------------------
    Returns:
        image_thresh: The image with classified edge pixels (weak or strong)
    """
    h, w = image.shape

    mag_max = np.max(grad_mag)  # Maximum gradient magnitude
    strong_thres = mag_max * strong_ratio
    weak_thres = strong_thres * weak_ratio

    image_thresh = np.zeros((h, w), dtype=np.int32)

    strong_i, strong_j = np.where(image >= strong_thres)
    weak_i, weak_j = np.where((image >= weak_thres) & (image <= strong_thres))

    image_thresh[strong_i, strong_j] = strong_pix
    image_thresh[weak_i, weak_j] = weak_pix

    return image_thresh

def hysteresis(image: np.ndarray, 
               weak_pix: int, 
               strong_pix: int) -> np.ndarray:
    """
    Apply edge tracking using hysteresis to finalize edge detection

    Parameters:
        image: The input image (edge classification result)
        weak_pix: The pixel value representing weak edges
        strong_pix: The pixel value representing strong edges

    --------------------------------------------------
    Returns:
        edges: The final edge-detected image after hysteresis
    """
    h, w = image.shape

    edges = np.copy(image)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            try:
                # Check neighbors for strong pixels
                if (image[i, j] == weak_pix):
                    if (image[i + 1, j - 1] == strong_pix) or \
                        (image[i + 1, j] == strong_pix) or \
                        (image[i + 1, j + 1] == strong_pix) or \
                        (image[i, j - 1] == strong_pix) or \
                        (image[i, j + 1] == strong_pix) or \
                        (image[i - 1, j - 1] == strong_pix) or \
                        (image[i - 1, j] == strong_pix) or \
                        (image[i - 1, j + 1] == strong_pix):

                        edges[i, j] = strong_pix
                    else:
                        edges[i, j] = 0
            except:
                pass

    return edges

def canny_edge_detection(image: np.ndarray, 
                         weak_pix: int = 100, 
                         strong_pix: int = 255, 
                         strong_ratio: float = 0.17, 
                         weak_ratio: float = 0.09) -> np.ndarray:
    """
    Perform Canny edge detection on the input image

    Parameters:
        image: The input image in RGB format (3-channel color image)
        weak_pix: The pixel value for weak edges
        strong_pix: The pixel value for strong edges
        strong_ratio: The ratio to determine the strong threshold
        weak_ratio: The ratio to determine the weak threshold

    --------------------------------------------------
    Returns:
        canny_edges: The final edge-detected image in binary format
    """
    image_blurred = gaussian_filter(image)  # Apply Gaussian filter to reduce noise

    grad_mag, theta = sobel_filters(image_blurred)  # Compute gradients

    non_max_img = non_maximum_suppress(grad_mag, theta)  # Apply non-maximum suppression

    image_thresh = double_thres(non_max_img, 
                                grad_mag, 
                                weak_pix, 
                                strong_pix, 
                                weak_ratio, 
                                strong_ratio)  # Apply double thresholding

    canny_edges = hysteresis(image_thresh, 
                             weak_pix, 
                             strong_pix)  # Perform hysteresis to finalize edges
    
    return canny_edges
