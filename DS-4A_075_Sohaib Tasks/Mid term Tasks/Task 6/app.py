# ============================================================
#  Task 6 – Animal Herd Detection with Map Alert
#  Course  : Programming for AI
#  Stack   : Flask + OpenCV + YOLOv4-tiny + Leaflet/OpenStreetMap
# ============================================================

import os
import json
import time
import base64
import random

from flask import (Flask, render_template, request,
                   jsonify, Response)
import cv2
import numpy as np

from detector import HerdDetector, download_progress

# ---------- app setup ----------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# one shared detector instance (loads model once)
detector = HerdDetector()

# ---------------------------------------------------------------
# Simulated GPS coordinates for demo mode
# (in a real deployment these would come from the camera's GPS)
# ---------------------------------------------------------------
DEMO_LOCATIONS = [
    {"name": "Serengeti Plains",    "lat": -2.333,  "lng": 34.833},
    {"name": "Kruger National Park","lat": -23.988, "lng": 31.554},
    {"name": "Yellowstone",         "lat": 44.428,  "lng": -110.588},
    {"name": "Amazon Basin",        "lat": -3.465,  "lng": -62.215},
    {"name": "Sahara Border",       "lat": 18.123,  "lng": 2.456},
]

# ---------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')


@app.route('/detect_image', methods=['POST'])
def detect_image():
    """
    Receives an uploaded image, runs herd detection,
    returns JSON with detections + annotated image (base64).
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Read image bytes directly into OpenCV (no disk write needed)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # ----- run detection -----
    detections, annotated_frame, herd_alert = detector.detect(frame)

    # Encode annotated image to base64 so we can send it in JSON
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Pick a random location for the map alert (demo GPS)
    location = random.choice(DEMO_LOCATIONS)
    # Add small random offset so repeated uploads look different
    location = {
        "name": location["name"],
        "lat":  round(location["lat"]  + random.uniform(-0.05, 0.05), 4),
        "lng":  round(location["lng"]  + random.uniform(-0.05, 0.05), 4),
    }

    return jsonify({
        'detections':  detections,
        'herd_alert':  herd_alert,
        'location':    location,
        'image_b64':   img_b64,
        'total_count': len(detections),
    })


@app.route('/detect_demo', methods=['GET'])
def detect_demo():
    """
    Returns a simulated detection result so students can test
    the UI without needing a real camera or YOLO weights file.
    """
    demo_animals = ['cow', 'sheep', 'horse', 'elephant', 'zebra', 'giraffe']
    count = random.randint(3, 9)
    detections = []
    for _ in range(count):
        detections.append({
            'label':      random.choice(demo_animals),
            'confidence': round(random.uniform(0.62, 0.97), 2),
            'box':        [random.randint(50, 400), random.randint(50, 300),
                           random.randint(80, 160), random.randint(80, 130)],
        })

    herd_alert = count >= 3
    location   = random.choice(DEMO_LOCATIONS)
    location   = {
        "name": location["name"],
        "lat":  round(location["lat"]  + random.uniform(-0.1, 0.1), 4),
        "lng":  round(location["lng"]  + random.uniform(-0.1, 0.1), 4),
    }

    return jsonify({
        'detections':  detections,
        'herd_alert':  herd_alert,
        'location':    location,
        'image_b64':   None,
        'total_count': count,
        'demo_mode':   True,
    })


@app.route('/video_feed')
def video_feed():
    """
    Live webcam stream with MJPEG multipart response.
    Each frame is processed by the herd detector.
    """
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def _generate_frames():
    """Generator that yields annotated JPEG frames from the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # yield a placeholder frame if no camera found
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, 'No Camera Found', (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 100), 2)
        _, buf = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _, annotated, _ = detector.detect(frame)
        _, buf = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')

    cap.release()


@app.route('/model_status')
def model_status():
    """Tells the front-end whether YOLO weights are loaded."""
    return jsonify({'yolo_loaded': detector.yolo_ready})


@app.route('/download_progress')
def get_download_progress():
    """
    Returns the current download progress for all 3 model files.
    The front-end polls this every second while downloading.
    """
    return jsonify(download_progress)


# ---------------------------------------------------------------
if __name__ == '__main__':
    print("\n Animal Herd Detection System")
    print(" Open  http://127.0.0.1:5000  in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
