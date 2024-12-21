import numpy as np 
from scipy import ndimage

# Using built-in function
#-----------------------------------------------------#
#def canny_edge_detection(img):                       |
#    canny_edges = cv2.Canny(img, 100, 150)           |
#    return canny_edges                               |
#-----------------------------------------------------#


# Converts an RGB image into grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Noise reduction using Gaussian filter
def gaussian_filter(image):
    # Create Gaussian kernel
    size = 5
    sigma = 1.4
    
    size = size // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * (sigma ** 2))
    gaussian_kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))) * normal
    
    gray_image = rgb2gray(image)
    
    image_blurred = ndimage.convolve(gray_image, gaussian_kernel)
    return image_blurred

# Gradient calculation using the Sobel-Feldman kernels convolution
def sobel_filters(image):
    # Sobel kernels
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32)
    
    # Derivatives Ix and Iy with respect to x and y axis
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)
    
    # Calculate gradient magnitudes and angle
    grad_mag = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    
    return (grad_mag, theta)

# Non-Maximum Suppression
def non_maximum_suppress(grad_mag, theta):
    h, w = grad_mag.shape
    non_max_img = np.zeros((h, w), dtype=np.int32)
    
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    # Looping through every pixel of the grayscale 
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


# Double Thresholding
def double_thres(image, grad_mag, weak_pix, strong_pix, weak_ratio, strong_ratio):
    h, w = image.shape

    mag_max = np.max(grad_mag)
    strong_thres = mag_max * strong_ratio
    weak_thres = strong_thres * weak_ratio

    image_thresh = np.zeros((h, w), dtype=np.int32)

    strong_i, strong_j = np.where(image >= strong_thres)
    weak_i, weak_j = np.where((image >= weak_thres) & (image <= strong_thres))

    image_thresh[strong_i, strong_j] = strong_pix
    image_thresh[weak_i, weak_j] = weak_pix

    return image_thresh

# Edge Tracking using Hysteresis
def hysteresis(image, weak_pix, strong_pix):
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

def canny_edge_detection(image, weak_pix=100, strong_pix=255, strong_ratio=0.17, weak_ratio=0.09): 
    image_blurred = gaussian_filter(image)

    grad_mag, theta = sobel_filters(image_blurred)

    non_max_img = non_maximum_suppress(grad_mag, theta)

    image_thresh = double_thres(non_max_img, grad_mag, weak_pix, strong_pix, weak_ratio, strong_ratio)

    canny_edges = hysteresis(image_thresh, weak_pix, strong_pix)
    
    return canny_edges
