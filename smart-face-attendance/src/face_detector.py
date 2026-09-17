"""
face_detector.py
Wraps OpenCV's Haar-cascade detector behind a small, testable interface.
Kept separate from face_recognizer.py so the detection algorithm can be
swapped (e.g. for a DNN detector) without touching recognition code
(single-responsibility / modular design).
"""

from typing import List, Tuple

import cv2
import numpy as np

import config


class FaceDetector:
    def __init__(self, cascade_path: str = config.HAAR_CASCADE_PATH):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return a list of (x, y, w, h) bounding boxes for faces found in a grayscale image."""
        faces = self.cascade.detectMultiScale(
            gray_image,
            scaleFactor=config.DETECTION_SCALE_FACTOR,
            minNeighbors=config.DETECTION_MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE,
        )
        return [tuple(f) for f in faces]

    def crop_faces(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """Convenience method: detect + return cropped face patches, resized to FACE_SIZE."""
        crops = []
        for (x, y, w, h) in self.detect(gray_image):
            face = gray_image[y : y + h, x : x + w]
            face = cv2.resize(face, config.FACE_SIZE, interpolation=cv2.INTER_AREA)
            crops.append(face)
        return crops
