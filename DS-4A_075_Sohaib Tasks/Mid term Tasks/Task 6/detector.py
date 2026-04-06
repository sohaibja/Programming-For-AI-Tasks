# ============================================================
#  detector.py  –  Animal Herd Detection Logic
#  Uses YOLOv4-tiny when weights are available,
#  falls back to a colour-contour approach otherwise.
# ============================================================

import os
import math
import cv2
import numpy as np

# ---------------------------------------------------------------
#  COCO class IDs that are animals we care about
#  (These are the indices in coco.names)
# ---------------------------------------------------------------
ANIMAL_CLASS_IDS = {
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
}

# ---------------------------------------------------------------
#  Colour palette per animal (BGR)
# ---------------------------------------------------------------
COLOURS = {
    'bird':     (255, 220,  50),
    'cat':      (255, 100, 100),
    'dog':      (100, 200, 255),
    'horse':    (180,  80, 255),
    'sheep':    (150, 255, 150),
    'cow':      ( 50, 200, 255),
    'elephant': (200, 150,  50),
    'bear':     ( 80,  80, 255),
    'zebra':    (255, 255, 255),
    'giraffe':  ( 50, 200, 200),
    'animal':   (100, 255, 100),   # fallback colour
}

# How many animals make a "herd"
HERD_THRESHOLD = 3

# Paths to YOLO model files
#   Download from: https://github.com/AlexeyAB/darknet/releases
MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'models')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'yolov4-tiny.weights')
CFG_PATH     = os.path.join(MODEL_DIR, 'yolov4-tiny.cfg')
NAMES_PATH   = os.path.join(MODEL_DIR, 'coco.names')


# ---------------------------------------------------------------
#  ALL 3 model files are auto-downloaded if missing.
#  weights = 23 MB  (progress is tracked and exposed to Flask)
# ---------------------------------------------------------------
AUTO_DOWNLOAD = {
    WEIGHTS_PATH: 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
    CFG_PATH:     'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
    NAMES_PATH:   'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names',
}

# Global dict that Flask can read to report progress to the browser
# Keys: filename string
# Values: dict with keys  pct, mb_done, mb_total, done, error
download_progress = {}


def _download_file(url, dest_path):
    """
    Download a file and save it to dest_path.
    Updates download_progress so the browser can poll for live status.
    """
    import urllib.request

    filename = os.path.basename(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Initialise progress entry
    download_progress[filename] = {
        'pct': 0, 'mb_done': 0, 'mb_total': 0,
        'done': False, 'error': False
    }

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct        = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb_done    = downloaded   / (1024 * 1024)
        mb_total   = total_size   / (1024 * 1024)

        download_progress[filename]['pct']      = round(pct, 1)
        download_progress[filename]['mb_done']  = round(mb_done, 1)
        download_progress[filename]['mb_total'] = round(mb_total, 1)

        # Terminal bar
        filled = int(pct / 5)
        bar    = '█' * filled + '░' * (20 - filled)
        print(f"\r[Detector] {filename}  [{bar}]  "
              f"{mb_done:.1f}/{mb_total:.1f} MB  {pct:.0f}%",
              end='', flush=True)

    print(f"[Detector] Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print()
        print(f"[Detector] Saved → {dest_path}")
        download_progress[filename]['pct']  = 100
        download_progress[filename]['done'] = True
        return True
    except Exception as e:
        print(f"\n[Detector] Download failed for {filename}: {e}")
        download_progress[filename]['error'] = True
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def ensure_config_files():
    """
    Check all 3 YOLO files and download any that are missing.
    Called automatically when HerdDetector is created.
    """
    for path, url in AUTO_DOWNLOAD.items():
        if not os.path.exists(path):
            _download_file(url, path)
        else:
            # Already on disk — mark as done so the UI shows green ticks
            filename = os.path.basename(path)
            download_progress[filename] = {
                'pct': 100, 'mb_done': 0, 'mb_total': 0,
                'done': True, 'error': False
            }


# ---------------------------------------------------------------
class HerdDetector:
    """
    Loads YOLOv4-tiny if the model files are present.
    Otherwise uses a simple contour-based placeholder that still
    produces bounding boxes so the rest of the pipeline works.
    """

    def __init__(self):
        self.net        = None
        self.classes    = []
        self.yolo_ready = False
        ensure_config_files()   # auto-download cfg + coco.names
        self._try_load_yolo()

    # ----------------------------------------------------------
    def _try_load_yolo(self):
        """Try to load YOLO weights. Silently skip if not found."""
        if (os.path.exists(WEIGHTS_PATH) and
                os.path.exists(CFG_PATH) and
                os.path.exists(NAMES_PATH)):
            try:
                self.net = cv2.dnn.readNet(WEIGHTS_PATH, CFG_PATH)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                with open(NAMES_PATH) as f:
                    self.classes = [line.strip() for line in f.readlines()]

                self.yolo_ready = True
                print("[Detector] YOLOv4-tiny loaded successfully.")
            except Exception as e:
                print(f"[Detector] Could not load YOLO: {e}")
        else:
            print("[Detector] YOLO weights not found – running in DEMO mode.")
            print(f"           Place yolov4-tiny.weights / .cfg / coco.names in: {MODEL_DIR}")

    # ----------------------------------------------------------
    def detect(self, frame):
        """
        Main detection entry point.
        Returns:
            detections  – list of dicts {label, confidence, box}
            annotated   – frame with bounding boxes drawn on it
            herd_alert  – True if HERD_THRESHOLD or more animals found
        """
        if self.yolo_ready:
            detections = self._yolo_detect(frame)
        else:
            detections = self._simple_detect(frame)

        herd_alert = len(detections) >= HERD_THRESHOLD
        annotated  = self._draw(frame.copy(), detections, herd_alert)

        return detections, annotated, herd_alert

    # ----------------------------------------------------------
    def _yolo_detect(self, frame):
        """Run YOLOv4-tiny and return animal detections only."""
        h, w = frame.shape[:2]

        # Build a blob from the frame
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255.0,
            size=(416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Get only the output layer names we need
        out_layers = self._get_output_layers()
        outputs    = self.net.forward(out_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores     = detection[5:]
                class_id   = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # Only keep animal classes with good confidence
                if class_id in ANIMAL_CLASS_IDS and confidence > 0.40:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    x  = cx - bw // 2
                    y  = cy - bh // 2

                    boxes.append([x, y, bw, bh])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Non-maximum suppression removes overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.40, 0.45)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                label = ANIMAL_CLASS_IDS.get(class_ids[i], 'animal')
                detections.append({
                    'label':      label,
                    'confidence': round(confidences[i], 2),
                    'box':        boxes[i],
                })

        return detections

    # ----------------------------------------------------------
    def _get_output_layers(self):
        """Get only the unconnected output layer names from YOLO."""
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # Handle both old and new OpenCV DNN API shapes
        if unconnected.ndim == 2:
            return [layer_names[i[0] - 1] for i in unconnected]
        else:
            return [layer_names[i - 1] for i in unconnected]

    # ----------------------------------------------------------
    def _simple_detect(self, frame):
        """
        Fallback detector that finds large moving blobs using
        background subtraction + contours.  Useful when YOLO
        weights are not available.  Returns generic 'animal' labels.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Only keep blobs of a plausible animal size
            if 1500 < area < 80000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'label':      'animal',
                    'confidence': round(0.55 + (area / 80000) * 0.30, 2),
                    'box':        [x, y, w, h],
                })

        return detections

    # ----------------------------------------------------------
    def _draw(self, frame, detections, herd_alert):
        """Draw bounding boxes, labels, and a herd alert banner."""
        h, w = frame.shape[:2]

        for det in detections:
            x, y, bw, bh = det['box']
            label        = det['label']
            conf         = det['confidence']
            colour       = COLOURS.get(label, COLOURS['animal'])

            # Draw filled semi-transparent rectangle behind text
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)
            text = f"{label}  {int(conf*100)}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), colour, -1)
            cv2.putText(frame, text, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Herd alert overlay at the top
        if herd_alert:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 42), (0, 60, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            msg = f"  HERD ALERT  {len(detections)} animals detected"
            cv2.putText(frame, msg, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 255, 100), 2)

        # Animal counter (bottom-right corner)
        counter_text = f"Animals: {len(detections)}"
        cv2.putText(frame, counter_text, (w - 180, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

        return frame

    # ----------------------------------------------------------
    @staticmethod
    def check_herd_proximity(detections, distance_threshold=150):
        """
        Checks if any animals are within distance_threshold pixels
        of each other (Euclidean distance between box centres).
        Returns True if a cluster of 3+ animals is found nearby.
        """
        if len(detections) < HERD_THRESHOLD:
            return False

        # Compute centre of each bounding box
        centres = []
        for det in detections:
            x, y, bw, bh = det['box']
            cx = x + bw // 2
            cy = y + bh // 2
            centres.append((cx, cy))

        # Count how many neighbours each animal has within the threshold
        for i, c1 in enumerate(centres):
            neighbour_count = 0
            for j, c2 in enumerate(centres):
                if i == j:
                    continue
                dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist < distance_threshold:
                    neighbour_count += 1
            if neighbour_count >= HERD_THRESHOLD - 1:
                return True

        return False
