"""
config.py
Central configuration for Smart Attendance & Wellbeing System.
All paths and tunable parameters live here so the rest of the codebase
never hard-codes a path or magic number.
"""

import os

# ---- Base paths -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
SAMPLE_DATASET_DIR = os.path.join(DATA_DIR, "sample_dataset")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RECOGNIZER_MODEL_PATH = os.path.join(MODELS_DIR, "lbph_recognizer.yml")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_svm.joblib")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
LOG_PATH = os.path.join(BASE_DIR, "app.log")

# ---- Face detection / recognition parameters -------------------------------
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(__import__("cv2").__file__), "data", "haarcascade_frontalface_default.xml"
)
FACE_SIZE = (200, 200)          # size faces are resized to before recognition
RECOGNITION_CONFIDENCE_THRESHOLD = 80.0  # LBPH: LOWER distance = more confident. Above -> "Unknown"
MIN_FACE_SIZE = (60, 60)
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5

# ---- Emotion model parameters ----------------------------------------------
EMOTION_LABELS = ["angry", "happy", "neutral", "sad", "surprise"]
EMOTION_IMAGE_SIZE = (48, 48)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
)

# ---- Misc -------------------------------------------------------------------
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

for _d in (DATA_DIR, KNOWN_FACES_DIR, SAMPLE_DATASET_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
