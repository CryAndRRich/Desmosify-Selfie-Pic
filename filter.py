import cv2
import numpy as np

def get_filtered(filename):
    img = cv2.imread(filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    gamma = 1.5
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    img = cv2.LUT(img, lookup_table)

    cv2.imwrite(filename, img)

