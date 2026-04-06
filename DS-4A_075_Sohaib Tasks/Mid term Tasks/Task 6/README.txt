============================================================
  Task 6 – Animal Herd Detection with Map Alert
  Course: Programming for AI
============================================================

PROJECT STRUCTURE
-----------------
task6/
├── app.py          ← Flask web server (routes + logic)
├── detector.py     ← Animal detection (YOLO / fallback)
├── requirements.txt
├── README.txt      ← This file
├── models/         ← Put YOLO weights here (see below)
│   ├── yolov4-tiny.weights
│   ├── yolov4-tiny.cfg
│   └── coco.names
└── templates/
    └── index.html  ← Full front-end dashboard


HOW TO RUN
----------
1. Install dependencies:
       pip install -r requirements.txt

2. (Optional) Download YOLO weights for real detection:
       https://github.com/AlexeyAB/darknet/releases
   Files needed:
       yolov4-tiny.weights  →  place in  models/
       yolov4-tiny.cfg      →  place in  models/
       coco.names           →  place in  models/

   Without the weights the app still works in DEMO mode
   (simulated detections + contour-based fallback on images).

3. Start the server:
       python app.py

4. Open your browser:
       http://127.0.0.1:5000


FEATURES
--------
✓ Upload any image and detect animals
✓ YOLO-based detection (15 animal classes from COCO)
✓ Fallback contour detector when YOLO weights absent
✓ Herd alert when 3+ animals detected (threshold adjustable)
✓ Live webcam stream with real-time detection
✓ OpenStreetMap (Leaflet.js) — FREE, no API key required
✓ Each detection drops a coloured pin on the world map
✓ Alert feed with timestamps and animal breakdown
✓ Night-vision themed dashboard UI
✓ Session statistics (total animals, herd alerts, scans)
✓ System log panel


ANIMAL CLASSES DETECTED (COCO dataset)
---------------------------------------
bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe


MAP / BONUS
-----------
Uses Leaflet.js + OpenStreetMap tiles — completely free,
no API key or account needed. Each detection event drops
a marker on the map with a popup showing:
  - Location name (simulated GPS in demo mode)
  - Animals detected and counts
  - Latitude / Longitude
  - Herd alert status

For real GPS integration, replace the DEMO_LOCATIONS list
in app.py with actual GPS coordinates from your camera.


STUDENT NOTES
-------------
- The code intentionally avoids advanced/complex patterns
  so it is easy to read and extend.
- Each file has clear section comments explaining the logic.
- The herd_proximity() function in detector.py shows how
  Euclidean distance is used to check animal clustering.
