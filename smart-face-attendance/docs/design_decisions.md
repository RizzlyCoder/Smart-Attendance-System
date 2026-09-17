# Design Decisions & Rationale

## 1. Why OpenCV LBPH instead of a deep-learning face embedding model (FaceNet/ArcFace)?

| Criterion | LBPH (chosen) | Deep embeddings (FaceNet/ArcFace) |
|---|---|---|
| Setup friction | `pip install opencv-contrib-python` only | Needs `dlib`/`torch`, often a C++ build toolchain, GPU recommended |
| Model size | None to download (trained on your own data) | 50–200 MB pretrained weights |
| Works with small enrollment sets (5–20 photos/person) | Yes, designed for this | Needs many images per identity or a strong pretrained backbone |
| CPU inference speed | Milliseconds | Seconds without GPU |
| Fits course syllabus (classical feature-based CV) | Yes | Overlaps more with deep learning courses |

Since the target use case is a small, controlled roster (a class or a small office) run entirely from the command line with no GPU guarantee, LBPH was the more appropriate choice. The trade-off is that LBPH is less robust to large pose/illumination variation than a deep embedding model — noted as a limitation and future enhancement.

## 2. Why HOG + SVM instead of a CNN for emotion recognition?

Same reasoning as above: HOG+SVM trains in seconds on CPU with `scikit-learn`/`scikit-image`, requires no GPU and no multi-hundred-MB pretrained weights, and is a textbook classical-CV feature pipeline (gradient histograms) that fits a Computer Vision course. A CNN trained on FER2013 would likely reach higher accuracy, and is listed under Future Enhancements as a drop-in replacement — `EmotionClassifier` already exposes a `train_from_directory()` / `predict()` interface that a CNN-based implementation could satisfy without changing any calling code (Enrollment, AttendanceSession, Analytics stay unchanged).

## 3. Why SQLite instead of a client-server database?

The project must be runnable by an evaluator with a single `pip install -r requirements.txt`. SQLite ships with Python's standard library (`sqlite3`), needs no server process, and is more than sufficient for a single-roster attendance log. The `Database` class isolates all SQL, so swapping to PostgreSQL/MySQL later only requires changing the connection layer.

## 4. Why a strict "one module, one responsibility" package layout?

`face_detector.py`, `face_recognizer.py`, and `emotion_model.py` are kept separate (rather than one big `vision.py`) so that:
- each has an independent, unit-testable interface,
- the detection algorithm can be swapped (e.g. to a DNN-based detector) without touching recognition or emotion code,
- it maps cleanly onto the three required functional modules (Enrollment, Attendance, Analytics), each of which composes the CV layer differently.

## 5. Known limitations

- LBPH and HOG+SVM are more sensitive to lighting/pose variation than deep models — acceptable for a controlled indoor camera, not for uncontrolled "in the wild" conditions.
- Attendance is capped at once per person per day by a DB uniqueness constraint; this is a design choice appropriate for attendance tracking, not a general-purpose recognition log.
- No liveness/anti-spoofing check — a printed photo could, in principle, be recognized. Out of scope for this project; noted under Future Enhancements.
