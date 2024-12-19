import cv2
import numpy as np

def luminance_adaption(image):
    # Retrieve the V channel from the HSV image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv_channels = list(cv2.split(hsv))
    V_comp = hsv_channels[2]

    # Prepare Gaussian kernels
    ker_size = 5
    gauss_ker_1 = cv2.getGaussianKernel(ker_size, 15)
    gauss_ker_2 = cv2.getGaussianKernel(ker_size, 80)
    gauss_ker_3 = cv2.getGaussianKernel(ker_size, 250)

    gauss_Vc_1 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_1)
    gauss_Vc_2 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_2)
    gauss_Vc_3 = cv2.filter2D(V_comp, cv2.CV_8U, gauss_ker_3)

    # Make LUT
    lut = [0] * 256
    for i in range(256):
        if i <= 127:
            lut[i] = 17.0 * (1.0 - np.sqrt(i / 127.0)) + 3.0
        else:
            lut[i] = 3.0 / 128.0 * (i - 127.0) + 3.0
        lut[i] = (-lut[i] + 20.0) / 17.0

    beta1 = cv2.LUT(gauss_Vc_1, np.array(lut, dtype=np.uint8))
    beta2 = cv2.LUT(gauss_Vc_2, np.array(lut, dtype=np.uint8))
    beta3 = cv2.LUT(gauss_Vc_3, np.array(lut, dtype=np.uint8))

    gauss_Vc_1 = gauss_Vc_1.astype(np.float64) / 255.0
    gauss_Vc_2 = gauss_Vc_2.astype(np.float64) / 255.0
    gauss_Vc_3 = gauss_Vc_3.astype(np.float64) / 255.0
    V_comp = V_comp.astype(np.float64) / 255.0

    V_comp = np.log(V_comp)
    gauss_Vc_1 = np.log(gauss_Vc_1)
    gauss_Vc_2 = np.log(gauss_Vc_2)
    gauss_Vc_3 = np.log(gauss_Vc_3)

    r = (3.0 * V_comp - beta1 * gauss_Vc_1 - beta2 * gauss_Vc_2 - beta3 * gauss_Vc_3) / 3.0
    R = np.exp(r)

    R_min, R_max = R.min(), R.max()
    V_w = (R - R_min) / (R_max - R_min)

    V_w = (V_w * 255.0).astype(np.uint8)

    # Compute histogram
    hist = cv2.calcHist([V_w], [0], None, [256], [0, 256])

    pdf = hist / (image.shape[0] * image.shape[1])

    pdf_min, pdf_max = pdf.min(), pdf.max()
    pdf = pdf_max * (pdf - pdf_min) / (pdf_max - pdf_min)

    cdf = np.cumsum(pdf)
    cdf[-1] = 1.0 - cdf[-2]

    V_w_max = V_w.max()
    for i in range(256):
        lut[i] = V_w_max * (i / V_w_max) ** (1.0 - cdf[i])

    V_out = cv2.LUT(V_w, np.array(lut, dtype=np.uint8))

    hsv_channels[2] = V_out
    hsv = cv2.merge(hsv_channels)

    image_output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)
    return image_output
