import os

import cv2
import numpy as np
import pytest

from src.db import Database
from src.enrollment import Enrollment, EnrollmentError
from tests.conftest import make_pattern_image


def test_enroll_rejects_empty_name(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    enrollment = Enrollment(db)
    with pytest.raises(EnrollmentError):
        enrollment.enroll_from_images("", ["some.jpg"])


def test_enroll_rejects_no_images(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    enrollment = Enrollment(db)
    with pytest.raises(EnrollmentError):
        enrollment.enroll_from_images("Alice", [])


def test_enroll_aborts_when_no_face_detected(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    enrollment = Enrollment(db)
    blank_path = str(tmp_config / "blank.jpg")
    cv2.imwrite(blank_path, np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(EnrollmentError):
        enrollment.enroll_from_images("Alice", [blank_path])


def test_enroll_success_with_mocked_detector(tmp_config, monkeypatch):
    """
    Haar-cascade face detection needs a real human face pattern, which we
    cannot ship as a bundled test asset. We mock crop_faces() so this test
    verifies enrollment's own logic (file saving + DB registration) in
    isolation from the third-party detector.
    """
    db = Database(db_path=str(tmp_config / "attendance.db"))
    enrollment = Enrollment(db)
    fake_face = make_pattern_image(1)
    monkeypatch.setattr(enrollment.detector, "crop_faces", lambda gray: [fake_face])

    img_path = str(tmp_config / "raw.jpg")
    cv2.imwrite(img_path, fake_face)

    saved = enrollment.enroll_from_images("Alice", [img_path])
    assert saved == 1
    assert db.get_user_by_name("Alice") is not None

    person_dir_files = os.listdir(enrollment._person_dir("Alice"))
    assert len(person_dir_files) == 1
