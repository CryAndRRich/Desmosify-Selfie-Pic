import numpy as np

# Algo for Basic Histogram Equalization
#---------------------------------------------------------------------------------------------#
#def histogram_equalization(image):                                                           |
#   hist = np.zeros((256,), np.uint8)                                                         |
#   h, w = image.shape[:2]                                                                    |
#   for i in range(h):                                                                        | 
#        for j in range(w):                                                                   | 
#            hist[image[i][j]] += 1                                                           |
#                                                                                             | 
#   hist = hist.ravel()                                                                       |
#                                                                                             |
#   cumulator = np.zeros_like(hist, np.float64)                                               |
#   for i in range(len(cumulator)):                                                           |
#       cumulator[i] = hist[:i].sum()                                                         |
#   new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255        |
#   new_hist = np.uint8(new_hist)                                                             |
#                                                                                             |
#   for i in range(h):                                                                        |
#       for j in range(w):                                                                    |
#           image[i,j] = new_hist[image[i,j]]                                                 | 
#                                                                                             |
#   return image                                                                              |
#---------------------------------------------------------------------------------------------#

# Algo for Contrast Limited Adaptive Histogram Equalization (CLAHE)

def clip_histogram(hist: np.ndarray, 
                   clip_limit: float, 
                   nr_x: int, 
                   nr_y: int, 
                   nr_bins: int) -> np.ndarray:
    """
    Clipping of the histogram and redistribution of bins

    Parameters:
        hist: The input histogram of shape (nr_x, nr_y, nr_bins)
        clip_limit: The maximum allowed value for histogram bins
        nr_x: The number of regions along the x-axis
        nr_y: The number of regions along the y-axis
        nr_bins: The number of histogram bins

    --------------------------------------------------
    Returns:
        hist: The clipped and redistributed histogram
    """
    # Clipping of the histogram and redistribution of bins
    for i in range(nr_x):
        for j in range(nr_y):
            nr_excess = 0
            for nr in range(nr_bins):
                excess = hist[i, j, nr] - clip_limit
                if excess > 0:
                    nr_excess += excess # Calculate total number of excess pixels
            
            bin_incr = nr_excess / nr_bins # Average bins increment
            upper = clip_limit - bin_incr
            for nr in range(nr_bins):
                if hist[i, j, nr] > clip_limit:
                    hist[i, j, nr] = clip_limit
                else:
                    if hist[i, j, nr] > upper: # High bins count
                        nr_excess += upper - hist[i, j, nr]
                        hist[i, j, nr] = clip_limit
                    else: # Low bins count
                        nr_excess -= bin_incr
                        hist[i, j, nr] += bin_incr
            
            if nr_excess > 0: # Redistribute remaining excess
                step_size = max(1, np.floor(1 + nr_excess / nr_bins))
                for nr in range(nr_bins):
                    nr_excess -= step_size
                    hist[i, j, nr] += step_size
                    if nr_excess < 1:
                        break
    
    return hist


def make_histogram(nr_x: int, 
                   nr_y: int, 
                   nr_bins: int, 
                   bins: np.ndarray, 
                   x_size: int, 
                   y_size: int) -> np.ndarray:
    """
    Create a histogram based on the given bins and region sizes

    Parameters:
        nr_x: The number of regions along the x-axis
        nr_y: The number of regions along the y-axis
        nr_bins: The number of histogram bins
        bins: The input bins
        x_size: The width of each region
        y_size: The height of each region

    --------------------------------------------------
    Returns:
        hist: The generated histogram of shape (nr_x, nr_y, nr_bins)
    """
    hist = np.zeros((nr_x, nr_y, nr_bins))
    for i in range(nr_x):
        for j in range(nr_y):
            bin_ = bins[i * x_size: (i + 1) * x_size, j * y_size:(j + 1) * y_size].astype(int)
            for i1 in range(x_size):
                for j1 in range(y_size):
                    hist[i, j, bin_[i1, j1]] += 1

    return hist


def map_histogram(nr_x: int, 
                  nr_y: int, 
                  nr_bins: int, 
                  hist: np.ndarray, 
                  nr_pix: int, 
                  max_val: int = 255, 
                  min_val: int = 0) -> np.ndarray:
    """
    Map histogram values to a specified range

    Parameters:
        nr_x: The number of regions along the x-axis
        nr_y: The number of regions along the y-axis
        nr_bins: The number of histogram bins
        hist: The input histogram of shape (nr_x, nr_y, nr_bins)
        nr_pix: The number of pixels in each region
        max_val: The maximum value for mapping
        min_val: The minimum value for mapping

    --------------------------------------------------
    Returns:
        map_: The mapped histogram of shape (nr_x, nr_y, nr_bins)
    """
    map_ = np.zeros((nr_x, nr_y, nr_bins))
    scale = (max_val - min_val) / float(nr_pix)
    for i in range(nr_x):
        for j in range(nr_y):
            sum_ = 0
            for nr in range(nr_bins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(min_val + sum_ * scale, max_val))

    return map_


def make_LUT(nr_bins: int, 
             max_val: int = 255, 
             min_val: int = 0) -> np.ndarray:
    """
    Create a look-up table for histogram mapping

    Parameters:
        nr_bins: The number of histogram bins
        max_val: The maximum value for mapping
        min_val: The minimum value for mapping

    --------------------------------------------------
    Returns:
        LUT: The look-up table (LUT) for mapping
    """
    bin_size = np.floor(1 + (max_val - min_val) / float(nr_bins))

    LUT = np.floor((np.arange(min_val, max_val + 1) - min_val) / float(bin_size))

    return LUT


def interpolate(sub_bin: np.ndarray, 
                UL: np.ndarray, 
                UR: np.ndarray, 
                BL: np.ndarray, 
                BR: np.ndarray, 
                sub_x: int, 
                sub_y: int) -> np.ndarray:
    """
    Interpolate sub-bins based on the neighboring bins

    Parameters:
        sub_bin: The sub-bin of the image to interpolate
        UL: The upper-left mapped values
        UR: The upper-right mapped values
        BL: The bottom-left mapped values
        BR: The bottom-right mapped values
        sub_x: The width of the sub-bin
        sub_y: The height of the sub-bin

    --------------------------------------------------
    Returns:
        sub_image: The interpolated sub-image
    """
    sub_image = np.zeros(sub_bin.shape)
    num = sub_x * sub_y
    for i in range(sub_x):
        inv_i = sub_x - i
        for j in range(sub_y):
            inv_j = sub_y - j
            val = sub_bin[i, j].astype(int)
            # Bilinear interpolation formula
            sub_image[i, j] = inv_i * (inv_j * UL[val] + j * UR[val]) + i * (inv_j * BL[val] + j * BR[val])
            # Normalize the interpolated value
            sub_image[i, j] = np.floor(sub_image[i, j] / float(num))

    return sub_image


def clahe_image(image: np.ndarray, 
                clip_limit: float, 
                nr_x: int = 0, 
                nr_y: int = 0, 
                nr_bins: int = 128) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image

    Parameters:
        image: The input image (2D or 3D array)
        clip_limit: The clipping limit for histogram bin values
        nr_x: The number of regions along the x-axis
        nr_y: The number of regions along the y-axis
        nr_bins: The number of histogram bins

    --------------------------------------------------
    Returns:
        new_image: The image after CLAHE processing
    """
    if clip_limit == 1:
        return

    h, w = image.shape
    nr_bins = max(nr_bins, 128)  # Ensure at least 128 bins for better granularity

    if nr_x == 0:
        # Default square block size of 8x8
        x_size, y_size = 8, 8
        nr_x = np.ceil(h / x_size).astype(int)  # Number of blocks along x-axis
        nr_y = np.ceil(w / y_size).astype(int)  # Number of blocks along y-axis

        # Handle any excess pixels
        excess_x = int(x_size * (nr_x - h / x_size))
        excess_y = int(y_size * (nr_y - w / y_size))

        # Append zeros to the image to match the required size
        if excess_x != 0:
            image = np.append(image, np.zeros((excess_x, image.shape[1])).astype(int), axis=0)
        if excess_y != 0:
            image = np.append(image, np.zeros((image.shape[0], excess_y)).astype(int), axis=1)
    else:
        x_size, y_size = round(h / nr_x), round(w / nr_y)  # Calculate region sizes

    nr_pix = x_size * y_size  # Number of pixels in each region
    new_image = np.zeros(image.shape)  # Output image initialized to zeros

    # Adjust clip_limit based on the region size
    clip_limit = max(1, clip_limit * x_size * y_size / nr_bins)

    LUT = make_LUT(nr_bins)  # Create the LUT (Look-Up Table)
    bins = LUT[image]  # Map the image pixels to the corresponding histogram bins

    hist = make_histogram(nr_x, nr_y, nr_bins, bins, x_size, y_size)  # Create the histogram
    hist = clip_histogram(hist, clip_limit, nr_x, nr_y, nr_bins)  # Clip the histogram

    map_ = map_histogram(nr_x, nr_y, nr_bins, hist, nr_pix)  # Map the histogram to the image

    xI = 0
    for i in range(nr_x + 1):
        if i == 0:
            sub_x = x_size // 2  # Top row: reduce the size of the first block
            xU, xB = 0, 0
        elif i == nr_x:
            sub_x = x_size // 2  # Bottom row: reduce the size of the last block
            xU, xB = nr_x - 1, nr_x - 1
        else:
            sub_x = x_size  # Default size for middle blocks
            xU, xB = i - 1, i

        yI = 0
        for j in range(nr_y + 1):
            if j == 0:
                sub_y = y_size // 2  # Left column: reduce the size of the first block
                yL, yR = 0, 0  
            elif j == nr_y:
                sub_y = y_size // 2  # Right column: reduce the size of the last block
                yL, yR = nr_y - 1, nr_y - 1
            else:
                sub_y = y_size  # Default size for middle blocks
                yL, yR = j - 1, j 

            # Get the pixel mappings for the four neighbors
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]

            # Extract the sub-bin (local region) from the image
            sub_bin = bins[xI:xI + sub_x, yI:yI + sub_y]
            sub_image = interpolate(sub_bin, UL, UR, BL, BR, sub_x, sub_y)  # Perform interpolation
            new_image[xI:xI + sub_x, yI:yI + sub_y] = sub_image  # Assign the interpolated values

            yI += sub_y  # Move to the next column

        xI += sub_x  # Move to the next row
    
    # Remove excess pixels if added earlier to ensure proper image size
    if excess_x == 0 and excess_y != 0:
        return new_image[:,:-excess_y]
    elif excess_x != 0 and excess_y == 0:
        return new_image[:-excess_x,:]
    elif excess_x != 0 and excess_y != 0:
        return new_image[:-excess_x,:-excess_y]
    else:
        return new_image

    
def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to an image using CLAHE for each color channel

    Parameters:
        image: The input image (3D array with shape (height, width, channels))

    --------------------------------------------------
    Returns:
        image_output: The image after histogram equalization (3D array with the same shape as input image)
    """
    image_output = image.copy()
    image_output[:,:,0] = clahe_image(image[:,:,0], 4, 0, 0)

    return image_output

