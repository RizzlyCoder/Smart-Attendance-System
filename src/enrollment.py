"""
enrollment.py
Functional Module 1: User Enrollment.
Captures face samples for a new person either from a webcam or from a
folder of existing images, and registers the person in the database.
"""

import os
from typing import List

import cv2

import config
from src.db import Database
from src.face_detector import FaceDetector
from src.utils.image_utils import equalize_histogram, load_image_grayscale, resize
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnrollmentError(Exception):
    pass


class Enrollment:
    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.detector = FaceDetector()

    def _person_dir(self, name: str) -> str:
        safe_name = "_".join(name.strip().split())
        return os.path.join(config.KNOWN_FACES_DIR, safe_name)

    def enroll_from_images(self, name: str, image_paths: List[str]) -> int:
        """Register `name` and save cropped, normalized face images from `image_paths`."""
        if not name or not name.strip():
            raise EnrollmentError("Name must not be empty.")
        if not image_paths:
            raise EnrollmentError("At least one image is required for enrollment.")

        person_dir = self._person_dir(name)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for idx, path in enumerate(image_paths):
            try:
                gray = load_image_grayscale(path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Skipping unreadable image %s: %s", path, e)
                continue
            faces = self.detector.crop_faces(gray)
            if not faces:
                logger.warning("No face detected in %s, skipping.", path)
                continue
            face = equalize_histogram(faces[0])
            out_path = os.path.join(person_dir, f"{saved:03d}.jpg")
            cv2.imwrite(out_path, face)
            saved += 1

        if saved == 0:
            raise EnrollmentError(
                "Could not detect a usable face in any provided image. Enrollment aborted."
            )

        existing = self.db.get_user_by_name(name)
        if existing is None:
            self.db.add_user(name)
            logger.info("Registered new user '%s' with %d face sample(s).", name, saved)
        else:
            logger.info("Added %d more face sample(s) for existing user '%s'.", saved, name)

        return saved

    def enroll_from_webcam(self, name: str, num_samples: int = 20, camera_index: int = 0) -> int:
        """Capture `num_samples` face images live from a webcam for enrollment."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise EnrollmentError(
                f"Could not open camera index {camera_index}. "
                "Use enroll_from_images() on a headless/CI machine instead."
            )

        person_dir = self._person_dir(name)
        os.makedirs(person_dir, exist_ok=True)
        saved = 0
        try:
            while saved < num_samples:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.crop_faces(gray)
                if faces:
                    face = equalize_histogram(faces[0])
                    out_path = os.path.join(person_dir, f"{saved:03d}.jpg")
                    cv2.imwrite(out_path, face)
                    saved += 1
                cv2.waitKey(100)
        finally:
            cap.release()

        if saved == 0:
            raise EnrollmentError("No face captured from webcam. Enrollment aborted.")

        if self.db.get_user_by_name(name) is None:
            self.db.add_user(name)
        logger.info("Captured %d webcam sample(s) for '%s'.", saved, name)
        return saved
