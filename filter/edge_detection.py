import cv2
import numpy as np

def canny_edge_detection(img):
    canny_edges = cv2.Canny(img, 100, 150)
    return canny_edges

def marr_hildreth_edge_detection(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    marr_hildreth_edges = cv2.convertScaleAbs(laplacian_abs)
    return marr_hildreth_edges

def edge_detection(img, methods):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edge_results = []
    for method in methods:
        if method == 'c':
            edge_results.append(canny_edge_detection(gray_image))
        if method == 'm':
            edge_results.append(marr_hildreth_edge_detection(gray_image))
    
    final_edge_result = sum(edge_results)
    return final_edge_result
            
