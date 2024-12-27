import numpy as np

from .canny_edge_detection import canny_edge_detection
from .marr_hildreth_edge_detection import marr_hildreth_edge_detection
from .sobel_edge_detection import sobel_edge_detection

def edge_detection(image, methods):
    edges = []
    
    # Implementing edge detection techniques and storing the output
    for method in methods:
        if method == 'c':
            image_output = canny_edge_detection(image)
        if method == 'm':
            image_output = marr_hildreth_edge_detection(image)
        if method == 's':
            image_output = sobel_edge_detection(image)

        edges.append(image_output)

    h, w = image.shape[:2]
    final_edge_result = np.zeros((h, w), dtype=np.uint8)
    
    # Looping through every pixel
    for i in range(h):
        for j in range(w):
            vote_count = sum([(edge[i, j] > 0) for edge in edges])
            
            # Majority Voting
            if vote_count >= len(methods) // 2 + 1:
                # If there are at least 2/3 methods mark it as an edge pixel,
                # it will be considered as an edge pixel
                final_edge_result[i, j] = 255 

    return final_edge_result
