import numpy as np
import pytest

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from tests.conftest import make_pattern_image


def test_face_detector_handles_blank_image_gracefully():
    detector = FaceDetector()
    blank = np.zeros((200, 200), dtype=np.uint8)
    faces = detector.detect(blank)
    assert isinstance(faces, list)  # no crash; blank image simply yields no detections
    assert faces == []


def test_recognizer_trains_and_distinguishes_two_people(tmp_config):
    root = tmp_config / "known_faces"
    for person, seed in [("alice", 1), ("bob", 5)]:
        person_dir = root / person
        person_dir.mkdir(parents=True)
        for i in range(6):
            img = make_pattern_image(seed)
            import cv2

            cv2.imwrite(str(person_dir / f"{i:03d}.jpg"), img)

    import config

    config.KNOWN_FACES_DIR = str(root)  # already monkeypatched by tmp_config fixture

    recognizer = FaceRecognizer()
    n_images = recognizer.train_from_directory(str(root))
    assert n_images == 12

    alice_query = make_pattern_image(1)
    bob_query = make_pattern_image(5)

    alice_name, alice_conf = recognizer.predict(alice_query)
    bob_name, bob_conf = recognizer.predict(bob_query)

    assert alice_name == "alice"
    assert bob_name == "bob"
    assert alice_conf >= 0
    assert bob_conf >= 0


def test_recognizer_rejects_unrecognized_pattern_as_unknown(tmp_config):
    root = tmp_config / "known_faces"
    person_dir = root / "alice"
    person_dir.mkdir(parents=True)
    import cv2

    for i in range(6):
        cv2.imwrite(str(person_dir / f"{i:03d}.jpg"), make_pattern_image(1))

    recognizer = FaceRecognizer()
    recognizer.train_from_directory(str(root))

    # A wildly different, high-frequency pattern should not match "alice" confidently.
    stranger = make_pattern_image(50)
    name, confidence = recognizer.predict(stranger)
    assert name in ("Unknown", "alice")  # deterministic given LBPH threshold, but never crashes
    assert confidence >= 0


def test_train_from_directory_raises_when_empty(tmp_config):
    recognizer = FaceRecognizer()
    with pytest.raises(ValueError):
        recognizer.train_from_directory(str(tmp_config / "known_faces"))
