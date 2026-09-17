"""
conftest.py
Shared pytest fixtures. Every test that touches the filesystem or the
database is redirected to a temporary directory via monkeypatched config
paths, so running the test suite never touches real enrollment data.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirects all persistent-storage paths used by config to a tmp dir."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "attendance.db"))
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", str(tmp_path / "known_faces"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "RECOGNIZER_MODEL_PATH", str(tmp_path / "models" / "lbph.yml"))
    monkeypatch.setattr(config, "LABEL_MAP_PATH", str(tmp_path / "models" / "labels.json"))
    monkeypatch.setattr(config, "EMOTION_MODEL_PATH", str(tmp_path / "models" / "emotion.joblib"))
    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    return tmp_path


def make_pattern_image(seed: int, size=(200, 200)) -> np.ndarray:
    """
    Deterministic, class-distinguishable synthetic grayscale image.
    Different seeds -> different stripe orientation/frequency, giving both
    LBPH and HOG features something genuinely separable to learn, without
    depending on any external face dataset.
    """
    rng = np.random.RandomState(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    freq = 0.05 + 0.02 * seed
    angle = seed * 0.7
    pattern = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    noise = rng.normal(0, 0.03, size=size)
    img = np.clip(pattern + noise, 0, 1) * 255
    return img.astype(np.uint8)


@pytest.fixture
def make_dataset_dir():
    """Factory fixture: build a <root>/<label>/*.jpg dataset from a {label: [seeds]} spec."""

    def _build(root, spec):
        for label, seeds in spec.items():
            label_dir = os.path.join(root, label)
            os.makedirs(label_dir, exist_ok=True)
            for i, seed in enumerate(seeds):
                img = make_pattern_image(seed)
                cv2.imwrite(os.path.join(label_dir, f"{i:03d}.jpg"), img)
        return root

    return _build
