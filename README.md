# Smart Attendance & Wellbeing System

A command-line Computer Vision system that enrolls people by face, marks attendance automatically via face recognition, and tags each check-in with a detected emotion — turning a plain attendance log into a lightweight wellbeing/analytics signal.

Built for the **Computer Vision** course "Build Your Own Project" evaluation.

---

## Overview

Three functional modules, chained through a single CLI:

1. **Enrollment** — register a new person from photos or a webcam.
2. **Attendance** — detect faces in an image/video/webcam feed, recognize identity, classify emotion, and log to a local database (once per person per day).
3. **Analytics & Reporting** — export CSV logs, per-person summaries, and attendance/emotion charts.

See `docs/architecture_diagram.png` and `docs/workflow_diagram.md` for the full system design, and `docs/design_decisions.md` for why each algorithm was chosen.

## Features

- Face detection with OpenCV Haar cascades
- Face recognition with OpenCV LBPH (no GPU, no external model download)
- Emotion classification with HOG features + linear SVM (5 classes: angry, happy, neutral, sad, surprise)
- SQLite-backed attendance log with duplicate-per-day prevention
- CSV export, per-person summary table, attendance trend chart, emotion distribution chart
- Fully CLI-driven — every feature is reachable from the terminal
- Structured logging to `app.log`
- Unit test suite (`pytest`) covering DB, recognition, emotion, analytics, and enrollment logic

## Technologies Used

- Python 3.10+
- OpenCV (`opencv-contrib-python`) — face detection & LBPH recognition
- scikit-image / scikit-learn — HOG feature extraction & SVM emotion classifier
- pandas / matplotlib — analytics & charting
- SQLite (via Python's built-in `sqlite3`)
- pytest — automated testing

---

## Project Structure

```
smart-face-attendance/
├── main.py                  # CLI entry point
├── config.py                # All paths & tunable parameters
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── db.py                 # SQLite data-access layer
│   ├── face_detector.py      # Haar cascade wrapper
│   ├── face_recognizer.py    # LBPH training/inference
│   ├── emotion_model.py      # HOG+SVM training/inference
│   ├── enrollment.py         # Module 1
│   ├── attendance.py         # Module 2
│   ├── analytics.py          # Module 3
│   └── utils/                # logger, image helpers
├── tests/                    # pytest unit tests + synthetic data fixtures
├── docs/                     # architecture/workflow/UML/ER diagrams + rationale
├── data/known_faces/         # enrolled face images (created at runtime)
├── models/                   # trained model artifacts (created at runtime)
└── reports/                  # generated CSV/chart output (created at runtime)
```

---

## Setup

### 1. Prerequisites
- Python 3.10 or newer
- A webcam is only needed for the live-camera enroll/attendance modes — everything also works from static image files, so a headless machine (e.g. a grading server) can run the entire pipeline.

### 2. Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the install

```bash
python main.py -h
```
You should see the list of available subcommands (`enroll`, `train-recognizer`, `train-emotion`, `attendance`, `report`, `list-users`).

---

## Usage

### Step 1 — Enroll people

From a folder of photos (works headless, recommended for grading):
```bash
python main.py enroll --name "Alice" --images "path/to/alice_photos/*.jpg"
```

From a webcam:
```bash
python main.py enroll --name "Alice" --samples 20
```
Repeat for each person you want the system to recognize.

### Step 2 — Train the face recognizer

```bash
python main.py train-recognizer
```
This reads every enrolled person's images from `data/known_faces/` and saves the trained model to `models/lbph_recognizer.yml`.

### Step 3 — Train the emotion classifier

Prepare an emotion dataset folder (see `docs/dataset_description.md` for the recommended FER2013-based layout):
```
emotion_dataset/
├── happy/*.jpg
├── sad/*.jpg
├── neutral/*.jpg
├── angry/*.jpg
└── surprise/*.jpg
```
Then run:
```bash
python main.py train-emotion --dataset path/to/emotion_dataset
```
> Emotion tagging is optional — attendance marking still works without this step; pass `--no-emotion` to `attendance` to skip it explicitly.

### Step 4 — Mark attendance

From a static image:
```bash
python main.py attendance --image path/to/photo.jpg
```

From a webcam (press `q` to stop):
```bash
python main.py attendance
```

### Step 5 — Generate reports

```bash
# CSV export
python main.py report --csv reports/attendance.csv

# Per-person summary table (printed to terminal)
python main.py report --summary

# Charts
python main.py report --trend-chart reports/trend.png --emotion-chart reports/emotion.png

# Filter by date range
python main.py report --csv reports/september.csv --start 2026-09-01 --end 2026-09-30
```

### Utility

```bash
python main.py list-users
```

---

## Testing

```bash
pip install pytest   # already in requirements.txt
pytest -v
```

The test suite uses a synthetic, seeded pattern-image generator (`tests/conftest.py`) so it runs end-to-end **without needing a webcam or any external dataset** — every module (database, LBPH recognizer, HOG+SVM emotion classifier, analytics, enrollment) is exercised with deterministic, reproducible data.

---

## Non-Functional Requirements Addressed

| Requirement | How it's addressed |
|---|---|
| **Performance** | LBPH + HOG/SVM run on CPU in milliseconds/seconds; no GPU or network calls needed at inference time |
| **Reliability** | Duplicate attendance prevented by a DB uniqueness constraint; emotion-model failures degrade gracefully instead of crashing attendance marking |
| **Usability** | Single CLI with `-h` help text on every subcommand; clear `[OK]`/`[ERROR]` output |
| **Maintainability** | Modular package layout (`src/`), one responsibility per file, all config centralized in `config.py` |
| **Error handling** | Custom exceptions (`EnrollmentError`, `AnalyticsError`) with descriptive messages instead of raw stack traces for expected failure cases |
| **Logging/monitoring** | Rotating file logger (`app.log`) + console output for every recognition/attendance/training event |
| **Scalability** | SQLite schema supports arbitrary numbers of users/records; swapping to a server DB only requires changing `db.py` |

## Future Enhancements

- Swap LBPH for a deep face-embedding model (e.g. ArcFace) for better robustness to pose/lighting, behind the same `FaceRecognizer` interface.
- Swap HOG+SVM for a CNN trained on full FER2013 for higher emotion accuracy.
- Add a `--train-test-split` evaluation flag reporting accuracy/confusion matrix for both models.
- Add liveness detection to prevent photo-based spoofing.
- Add a simple REST API wrapper around the existing modules for integration with a front-end dashboard.

## References

- Ahonen, T., Hadid, A., & Pietikäinen, M. (2006). *Face description with local binary patterns: Application to face recognition.*
- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection.*
- Viola, P., & Jones, M. (2001). *Rapid object detection using a boosted cascade of simple features.*
- Goodfellow, I. et al. (2013). *Challenges in representation learning: A report on three machine learning contests* (FER2013 dataset origin).
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/
- scikit-image HOG documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html
