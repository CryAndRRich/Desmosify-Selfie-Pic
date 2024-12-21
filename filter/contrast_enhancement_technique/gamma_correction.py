import numpy as np
import cv2

# Algo for Basic Gamma Correction
#-----------------------------------------------------------------------------#
#def gamma_correction(image, gamma = 1.0):                                    |
#   image = np.power(image, gamma)                                            |
#   max_val = np.max(image.ravel())                                           | 
#   image = image/max_val * 255                                               |
#   image = image.astype(np.uint8)                                            |
#                                                                             |
#   return image                                                              |
#-----------------------------------------------------------------------------#

# Algo for Improved Adaptive Gamma Correction with Weight Distribution (IAGCWD)

def iagcwd_image(image, gamma=0.5, truncated_cdf=False):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    unique_intensity = np.unique(image)

    prob_normalized = hist / hist.sum()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    
    # Weighting Distribution function is used to smooth the primary histogram
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** gamma)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** gamma))
    prob_normalized_wd = pn_temp / pn_temp.sum()
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
    
    inverse_cdf = 1 - cdf_prob_normalized_wd
    if truncated_cdf: 
        inverse_cdf = np.maximum(0.5, inverse_cdf)
    
    new_image = image.copy()
    for i in unique_intensity:
        new_image[image == i] = np.round(255 * (i / 255) ** inverse_cdf[i])
   
    return new_image

def process_dimmed_image(image):
    iagcwd = iagcwd_image(image, gamma=0.75, truncated_cdf=True)
    return iagcwd

def process_bright_image(image):
    img_negative = 255 - image
    iagcwd = 255 - iagcwd_image(img_negative, gamma=0.25, truncated_cdf=False)
    return iagcwd

def gamma_correction(image):
    # Extract intensity component of the image
    YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:,:,0]

    # Determine whether image is bright or dimmed
    threshold = 0.3
    exp_in = 112 # Expected global average intensity 
    h, w = image.shape[:2]
    mean_in = np.sum(Y / (h * w)) 
    
    # Process image for gamma correction
    image_output = None
    t = (mean_in - exp_in) / exp_in
    if t < -threshold: # Dimmed Image
        result = process_dimmed_image(Y)
        YCrCb[:,:,0] = result
        image_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    elif t > threshold: # Bright Image
        result = process_bright_image(Y)
        YCrCb[:,:,0] = result
        image_output = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    else:
        image_output = image
    
    return image_output


