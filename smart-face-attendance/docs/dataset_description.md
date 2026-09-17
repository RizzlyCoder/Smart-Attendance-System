# Dataset Description

This project uses **two independent, user-supplied datasets** rather than one fixed dataset, because it is a personalized enrollment system (it must recognize *specific* people, not generic face categories).

## 1. Face recognition dataset (self-collected, per deployment)

- **Source**: Captured by the user at enrollment time, via `python main.py enroll --name "Alice" --images "photos/*.jpg"` or via webcam (`--samples 20`).
- **Format**: `data/known_faces/<person_name>/000.jpg, 001.jpg, ...` — grayscale, face-cropped, histogram-equalized, resized to 200x200 by `Enrollment`.
- **Size**: Recommended 10–20 images per person, varying angle/expression/lighting slightly, for a robust LBPH model.
- **Labels**: The folder name (person's name) is the label; no manual labeling file needed.

## 2. Emotion recognition dataset (for training `EmotionClassifier`)

- **Recommended public dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (Kaggle), which provides 48x48 grayscale face crops labeled with 7 emotions.
- **Adaptation used here**: This project trains on a **reduced 5-class subset** (`angry, happy, neutral, sad, surprise` — see `config.EMOTION_LABELS`) reorganized into:
  ```
  <dataset_dir>/happy/*.jpg
  <dataset_dir>/sad/*.jpg
  <dataset_dir>/neutral/*.jpg
  <dataset_dir>/angry/*.jpg
  <dataset_dir>/surprise/*.jpg
  ```
  A short conversion note is in `README.md` under "Preparing the emotion dataset".
- **Why a subset of FER2013's 7 classes**: `disgust` and `fear` are heavily under-represented in FER2013 (a few hundred images vs. thousands for other classes) and are visually easy to confuse with `angry`/`sad` even for humans; dropping them gives a more balanced, more learnable classifier for a classical HOG+SVM pipeline.
- **For grading/demo purposes without downloading FER2013**: `tests/conftest.py` includes a synthetic pattern-image generator (`make_pattern_image`) used purely for automated unit tests, so the test suite runs without any external dataset dependency.

## Evaluation methodology

- **Face recognizer**: evaluated qualitatively via the CLI (`attendance --image <test_photo>`), and via unit tests (`tests/test_face_recognizer.py`) that check the model can correctly discriminate between distinct enrolled identities and rejects unseen "unknown" patterns via the confidence threshold in `config.RECOGNITION_CONFIDENCE_THRESHOLD`.
- **Emotion classifier**: `train_from_directory()` prints class distribution and training set size, which should be checked for class imbalance before trusting accuracy. For a real FER2013-based training run, we recommend an 80/20 train/test split and reporting accuracy + a confusion matrix (see "Future Enhancements" in the project report for adding this as a CLI flag).
