"""
attendance.py
Functional Module 2: Recognition-based Attendance + Wellbeing tagging.
Runs the trained face recognizer (and, optionally, the emotion classifier)
over a webcam stream or a static image/video file, and logs attendance to
the database exactly once per person per day.
"""

from datetime import datetime
from typing import List, Optional, Tuple

import cv2

import config
from src.db import Database
from src.emotion_model import EmotionClassifier
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.utils.image_utils import equalize_histogram, load_image_grayscale
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttendanceSession:
    def __init__(self, db: Database = None, use_emotion: bool = True):
        self.db = db or Database()
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.recognizer.load()
        self.use_emotion = use_emotion
        self.emotion_clf: Optional[EmotionClassifier] = None
        if use_emotion:
            try:
                self.emotion_clf = EmotionClassifier()
                self.emotion_clf.load()
            except FileNotFoundError:
                logger.warning("Emotion model not found -- continuing without emotion tagging.")
                self.use_emotion = False

    def _process_face(self, face_gray) -> Tuple[str, float, Optional[str]]:
        name, confidence = self.recognizer.predict(face_gray)
        emotion = None
        if self.use_emotion and name != "Unknown":
            try:
                emotion, _ = self.emotion_clf.predict(face_gray)
            except Exception as e:  # noqa: BLE001 - degrade gracefully, never crash attendance
                logger.warning("Emotion prediction failed: %s", e)
        return name, confidence, emotion

    def _log(self, name: str, confidence: float, emotion: Optional[str]) -> bool:
        if name == "Unknown":
            return False
        user = self.db.get_user_by_name(name)
        if user is None:
            logger.warning("Recognized name '%s' has no DB record; skipping log.", name)
            return False
        user_id = user[0]
        now = datetime.now()
        marked = self.db.mark_attendance(
            user_id,
            now.strftime(config.DATE_FORMAT),
            now.strftime(config.TIME_FORMAT),
            confidence,
            emotion,
        )
        if marked:
            logger.info("Marked attendance: %s (conf=%.1f, emotion=%s)", name, confidence, emotion)
        return marked

    # ------------------------------------------------------------------
    def run_on_image(self, image_path: str) -> List[Tuple[str, float, Optional[str], bool]]:
        """Process a single static image. Returns list of (name, confidence, emotion, was_marked)."""
        gray = load_image_grayscale(image_path)
        results = []
        for face in self.detector.crop_faces(gray):
            face = equalize_histogram(face)
            name, confidence, emotion = self._process_face(face)
            marked = self._log(name, confidence, emotion)
            results.append((name, confidence, emotion, marked))
        return results

    def run_on_webcam(self, camera_index: int = 0, max_frames: int = 300):
        """Run live recognition + attendance marking from a webcam until 'q' is pressed."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")
        frames = 0
        try:
            while frames < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in self.detector.detect(gray):
                    face = cv2.resize(gray[y : y + h, x : x + w], config.FACE_SIZE)
                    face = equalize_histogram(face)
                    name, confidence, emotion = self._process_face(face)
                    self._log(name, confidence, emotion)
                    label = f"{name} ({emotion})" if emotion else name
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                cv2.imshow("Attendance - press q to quit", frame)
                frames += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
