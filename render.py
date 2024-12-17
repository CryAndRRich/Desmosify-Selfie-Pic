from flask import Flask, json, request, jsonify, render_template
from flask_cors import CORS
import multiprocessing
import webbrowser
import cv2
import base64
import os
import potrace

from filter.gamma_correction import gamma_correction
from filter.histogram_equalization import histogram_equalization
from filter.edge_detection import edge_detection

from camera import get_selfie

app = Flask(__name__, template_folder='frontend')
CORS(app)
PORT = 5000

API_KEY = 'dcb31709b452b1cf9dc26972add0fda6'

SAMPLE_DIR = 'samples' 
RAW_SAMPLE_PATH = os.path.join('samples', 'sample0.png')
FILTERED_SAMPLE_PATH = os.path.join('samples', 'sample1.png')

FILE_EXT = 'png' 
COLOUR = '#2464b4' 
OPEN_BROWSER = True 
SHOW_GRID = True

sample = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock = False)
width = multiprocessing.Value('i', 0, lock = False)

def apply_filters(img, filters):
    if 'g' in filters:
        img = gamma_correction(img, gamma=1.5)
    if 'h' in filters:
        img = histogram_equalization(img)
    
    methods = []
    if 'c' in filters:
        methods.append('c')
    if 'm' in filters:
        methods.append('m')
    if 'p' in filters:
        methods.append('p')
    
    if len(methods) > 0:
        img = edge_detection(img, methods)

    return img

def get_trace(filename):
    image = cv2.imread(filename)
    edged = cv2.Canny(image, 30, 200)
    data = edged[::-1]
    bmp = potrace.Bitmap(data)
    path = bmp.trace(2, potrace.POTRACE_TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path


def get_latex(filename):
    latex = []
    path = get_trace(filename)

    for curve in path.curves:
        segments = curve.segments
        start = curve.start_point
        for segment in segments:
            x0, y0 = start.x, start.y
            if segment.is_corner:
                x1, y1 = segment.c.x, segment.c.y
                x2, y2 = segment.end_point.x, segment.end_point.y
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = segment.c1.x, segment.c1.y
                x2, y2 = segment.c2.x, segment.c2.y
                x3, y3 = segment.end_point.x, segment.end_point.y
                latex.append('((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),\
                (1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))' % \
                (x0, x1, x1, x2, x1, x2, x2, x3, y0, y1, y1, y2, y1, y2, y2, y3))
            start = segment.end_point
    return latex

def get_expressions(sample):
    exprid = 0
    exprs = {'latex': []}
    for expr in get_latex(SAMPLE_DIR + '/sample%d.%s' % (sample + 1, FILE_EXT)):
        exprs['latex'].append(expr)
        exprid += 1
    
    with open("sample_latex.json", "w") as file:
        json.dump(exprs, file, indent=4)
    
    return exprid

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/calculator')
def calculator():
    number_of_latex = get_expressions(0)

    return render_template('calculator.html', api_key=API_KEY, total_samples=len(os.listdir(SAMPLE_DIR)),
                           number_of_latex=number_of_latex, show_grid=SHOW_GRID)

@app.route('/apply-filters', methods=['POST'])
def apply_filters_route():
    filters = request.form.getlist('filters')

    image = cv2.imread(RAW_SAMPLE_PATH)

    if filters:
        filtered_image = apply_filters(image, filters)
        cv2.imwrite(FILTERED_SAMPLE_PATH, filtered_image)
        image = filtered_image

    _, buffer = cv2.imencode('.png', image)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image_url': f'data:image/png;base64,{encoded_img}'})

if __name__ == '__main__':
    #get_selfie()

    image = cv2.imread(RAW_SAMPLE_PATH)
    height.value = max(height.value, image.shape[0])
    width.value = max(width.value, image.shape[1])

    if OPEN_BROWSER:
        def open_browser():
            webbrowser.open('http://127.0.0.1:%d' % PORT)
        open_browser()

    app.run(host='127.0.0.1', port=PORT)
