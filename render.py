from flask import Flask, request, render_template
from flask_cors import CORS
import json
import potrace
import cv2
import multiprocessing
import os
import webbrowser

from camera import get_selfie
from filter import get_filtered

app = Flask(__name__, template_folder='frontend')
CORS(app)
PORT = 5000

API_KEY = 'dcb31709b452b1cf9dc26972add0fda6'
SAMPLE_DIR = 'samples' 
FILE_EXT = 'png' 
COLOUR = '#2464b4' 
SCREENSHOT_SIZE = [None, None] 
SCREENSHOT_FORMAT = 'png' 
OPEN_BROWSER = True 
SHOW_GRID = True

sample = multiprocessing.Value('i', 0)
height = multiprocessing.Value('i', 0, lock = False)
width = multiprocessing.Value('i', 0, lock = False)
sample_latex = 0

def get_contours(filename):
    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 30, 200)

    with sample.get_lock():
        sample.value += 1
        height.value = max(height.value, image.shape[0])
        width.value = max(width.value, image.shape[1])

    return edged[::-1]


def get_trace(data):
    bmp = potrace.Bitmap(data)
    path = bmp.trace(2, potrace.POTRACE_TURNPOLICY_MINORITY, 1.0, 1, .5)
    return path


def get_latex(filename):
    latex = []
    path = get_trace(get_contours(filename))

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
    exprs = []
    for expr in get_latex(SAMPLE_DIR + '/sample%d.%s' % (sample + 1, FILE_EXT)):
        exprid += 1
        exprs.append({'id': 'expr-' + str(exprid), 'latex': expr, 'color': COLOUR, 'secret': True})
    return exprs


@app.route('/')
def index():
    sample = int(request.args.get('sample'))
    if sample >= len(os.listdir(SAMPLE_DIR)):
        return {'result': None}

    return json.dumps({'result': sample_latex[sample] })


@app.route('/calculator')
def calculator():
    return render_template('calculator.html', 
                            api_key=API_KEY, height=height.value, 
                            width=width.value, total_samples=len(os.listdir(SAMPLE_DIR)), 
                            show_grid=SHOW_GRID, screenshot_size=SCREENSHOT_SIZE, 
                            screenshot_format=SCREENSHOT_FORMAT)

if __name__ == '__main__':
    print('First, take a selfie!')
    get_selfie()

    print('Start the pre-processing process...')
    filename = os.path.join('samples', 'sample1.png')
    get_filtered(filename)
    print('Done!')

    sample_latex = range(len(os.listdir(SAMPLE_DIR)))
    with multiprocessing.Pool(processes = multiprocessing.cpu_count()) as pool:
        sample_latex = pool.map(get_expressions, sample_latex)

        if OPEN_BROWSER:
            def open_browser():
                webbrowser.open('http://127.0.0.1:%d/calculator' % PORT)
            open_browser()

        app.run(host='127.0.0.1', port=PORT)
