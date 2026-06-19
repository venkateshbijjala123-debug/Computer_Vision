"""
config.py
---------
Centralized configuration for the Real-Time Emotion Detection System.

Keeping all tunable parameters in one place is standard practice in
production CV systems — it avoids "magic numbers" scattered through
the codebase and makes the system easy to retune without touching
core logic.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera / Capture settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                # default webcam
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FLIP_HORIZONTAL = True          # mirror view feels natural to users

# ---------------------------------------------------------------------------
# Face detection settings (OpenCV Haar Cascade is used for fast, lightweight
# face localization; DeepFace handles the actual emotion inference)
# ---------------------------------------------------------------------------
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACE_MIN_NEIGHBORS = 6
FACE_SCALE_FACTOR = 1.1
FACE_MIN_SIZE = (80, 80)

# ---------------------------------------------------------------------------
# Emotion inference settings
# ---------------------------------------------------------------------------
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

# DeepFace re-runs full emotion inference on every detected face. Running it
# on every single frame is unnecessary and slow (deep CNN forward pass),
# so we only run inference every N frames and reuse the last result in
# between -- a common optimization in real-time CV pipelines.
INFERENCE_EVERY_N_FRAMES = 5

# Confidence smoothing: how many recent predictions to average over so the
# displayed emotion doesn't flicker frame-to-frame.
SMOOTHING_WINDOW = 8

DEEPFACE_DETECTOR_BACKEND = "opencv"   # fast backend, paired with our own Haar pass
DEEPFACE_ENFORCE_DETECTION = False     # don't crash if DeepFace can't confirm a face

# ---------------------------------------------------------------------------
# UI / Rendering settings
# ---------------------------------------------------------------------------
FONT = "FONT_HERSHEY_SIMPLEX"
BOX_THICKNESS = 2
SHOW_FPS = True
SHOW_CONFIDENCE_BAR = True
RECORD_SESSION_LOG = True   # write emotion timeline to CSV for analysis

EMOTION_COLORS = {
    "angry":    (60, 60, 220),
    "disgust":  (60, 140, 60),
    "fear":     (180, 60, 180),
    "happy":    (60, 200, 80),
    "sad":      (200, 140, 60),
    "surprise": (60, 200, 220),
    "neutral":  (180, 180, 180),
}

WINDOW_NAME = "Real-Time Emotion Detection System"
