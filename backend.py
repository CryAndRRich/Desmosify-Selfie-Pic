from typing import List
from flask import Flask, json, request, jsonify, render_template
from flask_cors import CORS
import multiprocessing
import webbrowser
import cv2
import base64
import os
import numpy as np
import potrace

from filter import contrast_enhancement, edge_detection

from camera import get_selfie

app = Flask(__name__, template_folder='frontend')
CORS(app)
PORT = 5000

API_KEY = 'dcb31709b452b1cf9dc26972add0fda6'

SAMPLE_DIR = 'samples' 
IMAGE_LATEX_PATH = 'samples/selfie_latex.json'
RAW_IMAGE_PATH = os.path.join('samples', 'selfie.png')
FILTERED_IMAGE_PATH = os.path.join('samples', 'filtered-selfie.png')

FILE_EXT = 'png' 
COLOUR = '#2464b4' 
OPEN_BROWSER = True 

height = multiprocessing.Value('i', 0, lock = False)
width = multiprocessing.Value('i', 0, lock = False)
number_of_latex = 0

def apply_filters(image: np.ndarray, 
                  filters: str) -> np.ndarray:
    """
    Apply a series of filters to an image based on the provided filter string.

    Parameters:
        image: The input image to apply the filters to, as a 2D or 3D numpy array
        filters: A string of filter codes representing the operations to be applied
                'g', 'h', 'l' represent different enhancement techniques\n
                'c', 's', 'm' represent different edge detection methods

    --------------------------------------------------
    Returns:
        The processed image after applying the enhancement and edge detection filters
    """
    techniques = []
    methods = []

    # Categorize the filters into techniques and methods
    for char in filters:
        if char in ['g', 'h', 'l']:
            techniques.append(char)
        if char in ['c', 's', 'm']:
            methods.append(char)

    # Apply enhancement techniques if any
    if len(techniques) > 0:
        image = contrast_enhancement(image, techniques)

    # Apply edge detection methods if any
    if len(methods) > 0:
        image = edge_detection(image, methods)

    return image

def get_trace(filename: str) -> potrace.Path:
    """
    Generate a trace path from an image file

    Parameters:
        filename: The path to the image file

    --------------------------------------------------
    Returns:
        path: A potrace.Path object that contains the traced path from the image
    """
    image = cv2.imread(filename)
    edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = edges[::-1]  # Invert the image to match potrace's convention
    bmp = potrace.Bitmap(data)
    path = bmp.trace(2, potrace.POTRACE_TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path

def get_latex(filename: str) -> List[str]:
    """
    Generate LaTeX representation of the traced paths from an image.

    Parameters:
        filename: The path to the image file.

    --------------------------------------------------
    Returns:
        latex: A list of LaTeX expressions corresponding to the paths traced in the image.
    """
    latex = []
    path = get_trace(filename)

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start.x, start.y
            # Handle corner segments
            if segment.is_corner:
                x1, y1 = segment.c.x, segment.c.y
                x2, y2 = segment.end_point.x, segment.end_point.y
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            # Handle curve segments
            else:
                x1, y1 = segment.c1.x, segment.c1.y
                x2, y2 = segment.c2.x, segment.c2.y
                x3, y3 = segment.end_point.x, segment.end_point.y
                latex.append('((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),\
                (1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))' % \
                (x0, x1, x1, x2, x1, x2, x2, x3, y0, y1, y1, y2, y1, y2, y2, y3))
            start = segment.end_point

    return latex

def get_expressions(filename: str) -> None:
    """
    Generate LaTeX expressions from an image file and save them to a JSON file

    Parameters:
        filename: The name of the image file (without extension) to extract LaTeX expressions from
    """
    global number_of_latex 
    number_of_latex = 0
    exprs = {'latex': []}
    # Generate LaTeX expressions for the image
    for expr in get_latex(SAMPLE_DIR + '/%s.%s' % (filename, FILE_EXT)):
        exprs['latex'].append(expr)
        number_of_latex += 1
    
    # Save LaTeX expressions to a JSON file
    with open(IMAGE_LATEX_PATH, "w") as file:
        json.dump(exprs, file, indent=4)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/calculator')
def calculator():
    get_expressions('filtered-selfie')

    return render_template('calculator.html', api_key=API_KEY, height=height.value, width=width.value, 
                           number_of_latex=number_of_latex, color=COLOUR)

@app.route('/apply-filters', methods=['POST'])
def apply_filters_route():
    filters = request.form.getlist('filters')

    image = cv2.imread(RAW_IMAGE_PATH)

    if filters:
        filtered_image = apply_filters(image, filters)
        cv2.imwrite(FILTERED_IMAGE_PATH, filtered_image)
        image = filtered_image

    _, buffer = cv2.imencode('.png', image)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image_url': f'data:image/png;base64,{encoded_img}'})

@app.route('/desmos-render')
def desmos_render():
    with open(IMAGE_LATEX_PATH, "r") as file:
        data = json.load(file)
    
    return jsonify(data['latex'])

if __name__ == '__main__':
    print("""You already have a picture in the 'samples' folder, or you don't have one and want to take a photo with the computer's camera?
    Press 1 if you want to take a picture.
    Press 2 if you already have a picture, remember to rename it to 'selfie.png'
    """)
    
    num = int(input('Your choice: '))
    if num == 1:
        get_selfie(RAW_IMAGE_PATH)
        
    image = cv2.imread(RAW_IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot find or open the image. Please make sure the image's name is 'selfie.png'")
    height.value = max(height.value, image.shape[0])
    width.value = max(width.value, image.shape[1])

    if OPEN_BROWSER:
        def open_browser():
            webbrowser.open('http://127.0.0.1:%d' % PORT)
        open_browser()

    app.run(host='127.0.0.1', port=PORT)
