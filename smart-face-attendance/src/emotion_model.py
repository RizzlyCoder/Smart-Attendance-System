"""
emotion_model.py
Classical-CV emotion classifier: Histogram-of-Oriented-Gradients (HOG)
features + a linear-kernel Support Vector Machine. Chosen instead of a
deep CNN so the whole project trains/runs on CPU in seconds with no GPU
or multi-hundred-MB pretrained weights (see docs/design_decisions.md).

Expected training data layout:
    <dataset_dir>/<emotion_label>/*.jpg   e.g. dataset/happy/0001.jpg
Labels must be a subset of config.EMOTION_LABELS.
"""

import os
from typing import List, Tuple

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from src.utils.image_utils import list_image_files, load_image_grayscale, resize


def _extract_hog(face_gray: np.ndarray) -> np.ndarray:
    face_gray = resize(face_gray, config.EMOTION_IMAGE_SIZE)
    features = hog(
        face_gray,
        orientations=config.HOG_PARAMS["orientations"],
        pixels_per_cell=config.HOG_PARAMS["pixels_per_cell"],
        cells_per_block=config.HOG_PARAMS["cells_per_block"],
        feature_vector=True,
    )
    return features


class EmotionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="linear", probability=True, C=1.0)
        self._trained = False

    # ------------------------------------------------------------------
    def train_from_directory(self, dataset_dir: str) -> Tuple[int, List[str]]:
        X, y = [], []
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"No such directory: {dataset_dir}")

        labels_found = sorted(
            d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
        )
        if not labels_found:
            raise ValueError(f"No emotion class folders found under {dataset_dir}")

        for label in labels_found:
            for path in list_image_files(os.path.join(dataset_dir, label)):
                img = load_image_grayscale(path)
                X.append(_extract_hog(img))
                y.append(label)

        if len(set(y)) < 2:
            raise ValueError("Need images for at least 2 emotion classes to train.")

        X = np.array(X)
        X = self.scaler.fit_transform(X)
        self.svm.fit(X, y)
        self._trained = True
        self._save()
        return len(y), labels_found

    # ------------------------------------------------------------------
    def predict(self, face_gray: np.ndarray) -> Tuple[str, float]:
        if not self._trained:
            self.load()
        feat = _extract_hog(face_gray).reshape(1, -1)
        feat = self.scaler.transform(feat)
        proba = self.svm.predict_proba(feat)[0]
        idx = int(np.argmax(proba))
        return self.svm.classes_[idx], float(proba[idx])

    # ------------------------------------------------------------------
    def _save(self):
        joblib.dump({"scaler": self.scaler, "svm": self.svm}, config.EMOTION_MODEL_PATH)

    def load(self):
        if not os.path.exists(config.EMOTION_MODEL_PATH):
            raise FileNotFoundError(
                "No trained emotion model found. Run 'train-emotion' first."
            )
        bundle = joblib.load(config.EMOTION_MODEL_PATH)
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self._trained = True
