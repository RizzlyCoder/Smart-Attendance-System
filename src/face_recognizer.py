"""
face_recognizer.py
Trains and runs an OpenCV LBPH (Local Binary Patterns Histograms) face
recognizer. LBPH is chosen over deep-embedding approaches (e.g. FaceNet)
because it needs no GPU, no large pretrained weights download, and works
well for small, controlled enrollment sets such as a classroom/office
attendance roster -- see docs/design_decisions.md for the full rationale.
"""

import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from src.face_detector import FaceDetector
from src.utils.image_utils import equalize_histogram, list_image_files, load_image_grayscale, resize


class FaceRecognizer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.detector = FaceDetector()
        self.label_to_name: Dict[int, str] = {}
        self._trained = False

    # ------------------------------------------------------------------
    def train_from_directory(self, known_faces_dir: str = config.KNOWN_FACES_DIR) -> int:
        """
        Expects known_faces_dir/<person_name>/*.jpg structure.
        Returns the number of images used for training.
        """
        faces: List[np.ndarray] = []
        labels: List[int] = []
        self.label_to_name = {}
        next_label = 0
        total_images = 0

        if not os.path.isdir(known_faces_dir):
            raise FileNotFoundError(f"No such directory: {known_faces_dir}")

        person_dirs = sorted(
            d for d in os.listdir(known_faces_dir)
            if os.path.isdir(os.path.join(known_faces_dir, d))
        )
        if not person_dirs:
            raise ValueError(
                f"No enrolled persons found in {known_faces_dir}. Run 'enroll' first."
            )

        for person in person_dirs:
            person_dir = os.path.join(known_faces_dir, person)
            image_paths = list_image_files(person_dir)
            if not image_paths:
                continue
            self.label_to_name[next_label] = person
            for path in image_paths:
                img = load_image_grayscale(path)
                img = resize(img, config.FACE_SIZE)
                img = equalize_histogram(img)
                faces.append(img)
                labels.append(next_label)
                total_images += 1
            next_label += 1

        if not faces:
            raise ValueError("No usable face images found for training.")

        self.model.train(faces, np.array(labels))
        self._trained = True
        self._save()
        return total_images

    # ------------------------------------------------------------------
    def predict(self, face_gray: np.ndarray) -> Tuple[str, float]:
        """
        Returns (name, confidence). LBPH confidence is a DISTANCE: lower = more
        similar. Values above RECOGNITION_CONFIDENCE_THRESHOLD are labeled "Unknown".
        """
        if not self._trained:
            self.load()
        face_gray = resize(face_gray, config.FACE_SIZE)
        face_gray = equalize_histogram(face_gray)
        label, confidence = self.model.predict(face_gray)
        if confidence > config.RECOGNITION_CONFIDENCE_THRESHOLD:
            return "Unknown", confidence
        return self.label_to_name.get(label, "Unknown"), confidence

    # ------------------------------------------------------------------
    def _save(self):
        self.model.write(config.RECOGNIZER_MODEL_PATH)
        with open(config.LABEL_MAP_PATH, "w") as f:
            json.dump(self.label_to_name, f)

    def load(self):
        if not os.path.exists(config.RECOGNIZER_MODEL_PATH):
            raise FileNotFoundError(
                "No trained recognizer model found. Run 'train-recognizer' first."
            )
        self.model.read(config.RECOGNIZER_MODEL_PATH)
        with open(config.LABEL_MAP_PATH) as f:
            self.label_to_name = {int(k): v for k, v in json.load(f).items()}
        self._trained = True
