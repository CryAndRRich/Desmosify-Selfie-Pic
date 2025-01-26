import cv2
import numpy as np

def luminance_adaption(image: np.ndarray) -> np.ndarray:
    """
    Apply luminance adaptation to an image using Gaussian filtering and histogram equalization

    Parameters:
        image: The input image in BGR format (3-channel color image)

    --------------------------------------------------
    Returns:
        image_output: The image after luminance adaptation, in BGR format
    """
    # Convert image to HSV color space and extract the V (value) channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv_channels = list(cv2.split(hsv))
    V_comp = hsv_channels[2]  # Extract the V (value) component for luminance adaptation

    # Prepare Gaussian kernels with different standard deviations
    ker_size = 5
    gauss_ker_1 = cv2.getGaussianKernel(ker_size, 15)
    gauss_ker_2 = cv2.getGaussianKernel(ker_size, 80)
    gauss_ker_3 = cv2.getGaussianKernel(ker_size, 250)

    # Apply Gaussian filters to the V channel
    gauss_Vc_1 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_1)
    gauss_Vc_2 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_2)
    gauss_Vc_3 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_3)

    # Create Look-Up Table (LUT) for luminance adjustment
    lut = [0] * 256
    for i in range(256):
        if i <= 127:
            lut[i] = 17.0 * (1.0 - np.sqrt(i / 127.0)) + 3.0
        else:
            lut[i] = 3.0 / 128.0 * (i - 127.0) + 3.0
        lut[i] = (-lut[i] + 20.0) / 17.0

    # Apply LUT to Gaussian filtered images
    beta1 = cv2.LUT(gauss_Vc_1, np.array(lut, dtype=np.uint8))
    beta2 = cv2.LUT(gauss_Vc_2, np.array(lut, dtype=np.uint8))
    beta3 = cv2.LUT(gauss_Vc_3, np.array(lut, dtype=np.uint8))

    # Convert Gaussian filtered images to float for logarithmic processing
    gauss_Vc_1 = gauss_Vc_1.astype(np.float64) / 255.0
    gauss_Vc_2 = gauss_Vc_2.astype(np.float64) / 255.0
    gauss_Vc_3 = gauss_Vc_3.astype(np.float64) / 255.0
    V_comp = V_comp.astype(np.float64) / 255.0

    # Apply logarithmic transformation to the V channel and filtered images
    eps = 1e-9 # Epsilon to avoid log(0)
    V_comp = np.log(V_comp + eps)
    gauss_Vc_1 = np.log(gauss_Vc_1 + eps)
    gauss_Vc_2 = np.log(gauss_Vc_2 + eps)
    gauss_Vc_3 = np.log(gauss_Vc_3 + eps)

    # Compute the result of luminance adaptation
    r = (3.0 * V_comp - beta1 * gauss_Vc_1 - beta2 * gauss_Vc_2 - beta3 * gauss_Vc_3) / 3.0
    R = np.exp(r)

    # Normalize the result to the range [0, 255]
    R_min, R_max = R.min(), R.max()
    V_w = (R - R_min) / (R_max - R_min)

    V_w = (V_w * 255.0).astype(np.uint8)

    # Compute histogram and Probability Density Function (PDF)
    hist = cv2.calcHist([V_w], [0], None, [256], [0, 256])

    pdf = hist / (image.shape[0] * image.shape[1])  # Normalize the histogram

    pdf_min, pdf_max = pdf.min(), pdf.max()
    pdf = pdf_max * (pdf - pdf_min) / (pdf_max - pdf_min)  # Normalize PDF to [0, 1]

    # Compute Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pdf)
    cdf[-1] = 1.0 - cdf[-2]  # Ensure that the last value is 1

    # Update LUT based on the CDF
    V_w_max = V_w.max()
    for i in range(256):
        lut[i] = V_w_max * (i / V_w_max) ** (1.0 - cdf[i])

    # Apply the updated LUT to the value channel
    V_out = cv2.LUT(V_w, np.array(lut, dtype=np.uint8))

    # Replace the V channel with the adapted values
    hsv_channels[2] = V_out
    hsv = cv2.merge(hsv_channels)

    # Convert back to BGR color space and return the adapted image
    image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
    return image_output
