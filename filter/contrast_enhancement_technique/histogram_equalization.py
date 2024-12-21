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

def clip_histogram(hist, clip_limit, nr_x, nr_y, nr_bins):
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


def make_histogram(nr_x, nr_y, nr_bins, bins, x_size, y_size):
    hist = np.zeros((nr_x, nr_y, nr_bins))
    for i in range(nr_x):
        for j in range(nr_y):
            bin_ = bins[i * x_size: (i + 1) * x_size, j * y_size:(j + 1) * y_size].astype(int)
            for i1 in range(x_size):
                for j1 in range(y_size):
                    hist[i, j, bin_[i1, j1]] += 1
    
    return hist

def map_histogram(nr_x, nr_y, nr_bins, hist, nr_pix, max_val=255, min_val=0):
    map_ = np.zeros((nr_x, nr_y, nr_bins))
    scale = (max_val - min_val) / float(nr_pix)
    for i in range(nr_x):
        for j in range(nr_y):
            sum_ = 0
            for nr in range(nr_bins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(min_val + sum_ * scale, max_val))
    
    return map_

def make_LUT(nr_bins, max_val=255, min_val=0):
    # Speed up histogram clipping
    bin_size = np.floor(1 + (max_val - min_val) / float(nr_bins))

    LUT = np.floor((np.arange(min_val, max_val + 1) - min_val) / float(bin_size))

    return LUT

def interpolate(sub_bin, UL, UR, BL, BR, sub_x, sub_y):
    sub_image = np.zeros(sub_bin.shape)
    num = sub_x * sub_y
    for i in range(sub_x):
        inv_i = sub_x - i
        for j in range(sub_y):
            inv_j = sub_y - j
            val = sub_bin[i, j].astype(int)
            sub_image[i, j] = inv_i * (inv_j * UL[val] + j * UR[val]) + i * (inv_j * BL[val] + j * BR[val])
            sub_image[i, j] = np.floor(sub_image[i, j] / float(num))
    
    return sub_image

def clahe_image(image, clip_limit, nr_x=0, nr_y=0, nr_bins=128):
    if clip_limit == 1:
        return
    
    h, w = image.shape
    nr_bins = max(nr_bins, 128)

    if nr_x == 0:
        #Taking dimensions of each contextual region to be a square of 8x8
        x_size, y_size = 8, 8
        nr_x = np.ceil(h / x_size).astype(int)
        nr_y = np.ceil(w / y_size).astype(int)
        
        excess_x = int(x_size * (nr_x - h / x_size))
        excess_y = int(y_size * (nr_y - w / y_size))

        if excess_x != 0:
            image = np.append(image, np.zeros((excess_x, image.shape[1])).astype(int), axis=0)
        if excess_y != 0:
            image = np.append(image, np.zeros((image.shape[0], excess_y)).astype(int), axis=1)
    else:
        x_size, y_size = round(h / nr_x), round(w / nr_y)
    
    nr_pix = x_size*y_size
    new_image = np.zeros(image.shape)

    clip_limit = max(1, clip_limit * x_size * y_size / nr_bins)
    
    LUT = make_LUT(nr_bins)
    bins = LUT[image]

    hist = make_histogram(nr_x, nr_y, nr_bins, bins, x_size, y_size)
    hist = clip_histogram(hist, clip_limit, nr_x, nr_y, nr_bins)

    map_ = map_histogram(nr_x, nr_y, nr_bins, hist, nr_pix)

    xI = 0
    for i in range(nr_x + 1):
        if i == 0: # Special case: Top row
            sub_x = x_size // 2
            xU, xB = 0, 0
        elif i == nr_x: # Special case: Bottom row
            sub_x = x_size // 2
            xU, xB = nr_x - 1, nr_x - 1
        else: # Default values
            sub_x = x_size
            xU, xB = i - 1, i
        
        yI = 0
        for j in range(nr_y + 1):
            if j == 0: # Special case: Left column
                sub_y = y_size // 2
                yL, yR = 0, 0
            elif j == nr_y: # Special case: Right column
                sub_y = y_size // 2
                yL, yR = nr_y - 1, nr_y - 1
            else: # Default values
                sub_y = y_size
                yL, yR = j - 1, j 

            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]

            sub_bin = bins[xI:xI + sub_x, yI:yI + sub_y]
            sub_image = interpolate(sub_bin, UL, UR, BL, BR, sub_x, sub_y)
            new_image[xI:xI + sub_x, yI:yI + sub_y] = sub_image

            yI += sub_y

        xI += sub_x
    
    if excess_x == 0 and excess_y != 0:
        return new_image[:,:-excess_y]
    elif excess_x != 0 and excess_y == 0:
        return new_image[:-excess_x,:]
    elif excess_x != 0 and excess_y != 0:
        return new_image[:-excess_x,:-excess_y]
    else:
        return new_image
    
def histogram_equalization(image):
    image_output = image.copy()
    image_output[:,:,0] = clahe_image(image[:,:,0], 4, 0, 0)

    return image_output

