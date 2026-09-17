#!/usr/bin/env python3
"""
build_report.py
Generates the formal Project Report PDF (15 required sections) for the
Smart Attendance & Wellbeing System, pulling in the diagrams and terminal
screenshots already produced under docs/ and reports/.
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
REPORTS = os.path.join(ROOT, "reports")
OUT_PATH = os.path.join(ROOT, "Project_Report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", fontSize=26, leading=32, alignment=TA_CENTER, spaceAfter=20, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="CoverSub", fontSize=14, leading=20, alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#334155")))
styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceBefore=18, spaceAfter=10, fontName="Helvetica-Bold", textColor=colors.HexColor("#1E293B")))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold", textColor=colors.HexColor("#1E293B")))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, spaceAfter=8, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=14))

story = []


def h1(text):
    story.append(Paragraph(text, styles["H1"]))


def h2(text):
    story.append(Paragraph(text, styles["H2"]))


def body(text):
    story.append(Paragraph(text, styles["Body"]))


def bullets(items):
    story.append(
        ListFlowable(
            [ListItem(Paragraph(i, styles["Body"]), bulletColor=colors.HexColor("#1E293B")) for i in items],
            bulletType="bullet",
            leftIndent=16,
        )
    )
    story.append(Spacer(1, 6))


def figure(path, caption, width=15 * cm):
    if not os.path.exists(path):
        return
    img = Image(path)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    story.append(Paragraph(caption, styles["Caption"]))


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))


# ============================================================ COVER PAGE
story.append(Spacer(1, 4 * cm))
story.append(Paragraph("Smart Attendance & Wellbeing System", styles["CoverTitle"]))
story.append(Paragraph("Face Recognition-Based Attendance with Emotion Analytics", styles["CoverSub"]))
story.append(Spacer(1, 1 * cm))
story.append(Paragraph("Project Report", styles["CoverSub"]))
story.append(Spacer(1, 2 * cm))
table(
    [
        ["Course", "Computer Vision"],
        ["Submission Type", "Flipped Course Evaluation -- Build Your Own Project"],
        ["Repository", "https://github.com/<your-username>/<your-repo>"],
        ["Submission Date", "September 2026"],
    ],
    col_widths=[4.5 * cm, 10.5 * cm],
)
story.append(PageBreak())

# ============================================================ 1. Introduction
h1("2. Introduction")
body(
    "This report documents the design and implementation of a command-line Computer Vision "
    "system that automates attendance marking using face recognition and enriches that record "
    "with a lightweight, per-check-in emotion estimate. The project applies core Computer "
    "Vision techniques covered in the course -- Haar-cascade face detection, Local Binary "
    "Pattern Histogram (LBPH) face recognition, and Histogram-of-Oriented-Gradients (HOG) "
    "feature extraction with an SVM classifier -- inside a single, modular, fully "
    "terminal-operable application."
)
body(
    "The system is organized into three functional modules (Enrollment, Attendance, and "
    "Analytics/Reporting), backed by a SQLite database, and is accompanied by an automated "
    "test suite that validates each module using deterministic synthetic data."
)

# ============================================================ 2. Problem Statement
h1("3. Problem Statement")
body(
    "Manual attendance tracking (roll calls, sign-in sheets) is slow, easy to falsify, and "
    "provides no insight beyond a bare present/absent record. Small classrooms, workshops, or "
    "office teams need a lightweight way to confirm attendance using something harder to fake "
    "than a signature, and to get a rough, non-intrusive sense of group wellbeing over time, "
    "without adopting a heavyweight commercial biometric platform."
)
body(
    "This project addresses that need with a self-hosted, CLI-driven system: it enrolls a "
    "known roster of people by face, automatically recognizes and logs attendance from a photo "
    "or webcam feed, and tags each check-in with a coarse emotion label -- giving a facilitator "
    "both an attendance register and a rough wellbeing trend line."
)

# ============================================================ 3. Functional Requirements
h1("4. Functional Requirements")
body("The system implements three major functional modules:")
bullets(
    [
        "<b>Module 1 -- Enrollment:</b> Register a new person from a batch of photos or a live webcam "
        "session. Input: name + images/webcam stream. Output: normalized face images saved to "
        "<font face='Courier'>data/known_faces/&lt;name&gt;/</font> and a new row in the <font face='Courier'>users</font> table.",
        "<b>Module 2 -- Attendance:</b> Detect and recognize faces in an image, video frame, or webcam "
        "stream; classify emotion; log one attendance record per person per day. Input: image path "
        "or webcam feed. Output: console summary + a row in the <font face='Courier'>attendance</font> table.",
        "<b>Module 3 -- Analytics &amp; Reporting:</b> Export attendance data to CSV, compute a per-person "
        "summary (days present, dominant emotion), and render attendance-trend and emotion-distribution "
        "charts. Input: date range (optional). Output: CSV file(s) and PNG chart(s).",
    ]
)
body(
    "All three modules are reachable through a single CLI entry point (<font face='Courier'>main.py</font>), "
    "giving the system a clear, linear workflow: enroll &rarr; train &rarr; mark attendance &rarr; report."
)

# ============================================================ 4. Non-Functional Requirements
h1("5. Non-Functional Requirements")
table(
    [
        ["Requirement", "How it is addressed"],
        ["Performance", "LBPH and HOG+SVM run on CPU in milliseconds/seconds; no GPU or network calls at inference time."],
        ["Reliability", "A DB uniqueness constraint prevents duplicate attendance; emotion-model failures degrade gracefully instead of crashing attendance marking."],
        ["Usability", "Single CLI with -h help text on every subcommand; clear [OK]/[ERROR] console output."],
        ["Maintainability", "Modular package layout (src/), one responsibility per file, all configuration centralized in config.py."],
        ["Error Handling", "Custom exceptions (EnrollmentError, AnalyticsError) with descriptive messages instead of raw stack traces."],
        ["Logging / Monitoring", "Rotating file logger (app.log) plus console output for every recognition/attendance/training event."],
        ["Scalability", "SQLite schema supports an arbitrary number of users/records; swapping to a server DB only requires changing db.py."],
    ],
    col_widths=[3.7 * cm, 11.3 * cm],
)

# ============================================================ 5. System Architecture
h1("6. System Architecture")
body(
    "The system is layered into a CLI layer, three core functional modules, a Computer Vision "
    "layer (detection / recognition / emotion), and a storage layer, as shown below."
)
figure(os.path.join(DOCS, "architecture_diagram.png"), "Figure 1: System architecture")

# ============================================================ 6. Design Diagrams
story.append(PageBreak())
h1("7. Design Diagrams")

h2("7.1 Use Case Diagram")
figure(os.path.join(DOCS, "use_case_diagram.png"), "Figure 2: Use case diagram")

h2("7.2 Workflow / Process Flow Diagram")
figure(os.path.join(DOCS, "workflow_diagram.png"), "Figure 3: End-to-end workflow")

story.append(PageBreak())
h2("7.3 Sequence Diagram -- Attendance Marking Flow")
figure(os.path.join(DOCS, "sequence_diagram.png"), "Figure 4: Sequence diagram for a single attendance check")

h2("7.4 Class / Component Diagram")
figure(os.path.join(DOCS, "class_diagram.png"), "Figure 5: Class/component diagram")

story.append(PageBreak())
h2("7.5 ER Diagram / Schema Design")
figure(os.path.join(DOCS, "er_diagram.png"), "Figure 6: Entity-relationship diagram")
body(
    "<b>users</b>(user_id PK, name UNIQUE, created_at) &mdash; one row per enrolled person.<br/>"
    "<b>attendance</b>(record_id PK, user_id FK, date, time, confidence, emotion, "
    "UNIQUE(user_id, date)) &mdash; one row per person per day, enforced by the database itself."
)

# ============================================================ 7. Design Decisions & Rationale
h1("8. Design Decisions & Rationale")
h2("8.1 Face recognition: OpenCV LBPH vs. deep embeddings")
body(
    "LBPH was chosen over deep-embedding approaches (FaceNet/ArcFace) because the target "
    "deployment is a small, controlled roster (5-20 photos per person) run from the command "
    "line with no guaranteed GPU. LBPH needs no pretrained-weight download, trains in seconds "
    "on CPU, and is a classical feature-based technique central to this course's syllabus. The "
    "trade-off is reduced robustness to large pose/illumination variation compared to deep "
    "models -- acceptable for a controlled indoor camera, and listed under Future Enhancements."
)
h2("8.2 Emotion recognition: HOG + SVM vs. a CNN")
body(
    "Similarly, HOG+SVM trains in seconds on CPU using scikit-learn/scikit-image, with no GPU "
    "and no large pretrained model, making the whole project installable with a single "
    "<font face='Courier'>pip install -r requirements.txt</font>. The EmotionClassifier class exposes a stable "
    "<font face='Courier'>train_from_directory()</font> / <font face='Courier'>predict()</font> interface, so a future CNN-based "
    "implementation could be substituted without changing any calling code."
)
h2("8.3 Storage: SQLite vs. a client-server database")
body(
    "SQLite ships with Python's standard library, needs no server process, and is more than "
    "sufficient for a single-roster attendance log, keeping the setup to one pip install."
)

# ============================================================ 8. (folded into 7 per numbering below)
# Renumbering note: sections 8-15 follow the exact list required by the submission guidelines.

# ============================================================ 8. Implementation Details
story.append(PageBreak())
h1("9. Implementation Details")
body("The codebase follows a strict one-module-one-responsibility layout:")
table(
    [
        ["File", "Responsibility"],
        ["config.py", "Central configuration: all paths and tunable parameters"],
        ["src/db.py", "SQLite data-access layer (users, attendance)"],
        ["src/face_detector.py", "Haar-cascade face detection wrapper"],
        ["src/face_recognizer.py", "LBPH training / inference / persistence"],
        ["src/emotion_model.py", "HOG feature extraction + SVM training / inference"],
        ["src/enrollment.py", "Functional Module 1 -- enrollment from images or webcam"],
        ["src/attendance.py", "Functional Module 2 -- recognition + attendance + emotion logging"],
        ["src/analytics.py", "Functional Module 3 -- CSV export, summaries, charts"],
        ["src/utils/", "Logging and image-processing helper functions"],
        ["main.py", "argparse-based CLI tying every module together"],
        ["tests/", "pytest suite with a synthetic, seeded data generator"],
    ],
    col_widths=[4.5 * cm, 10.5 * cm],
)
body(
    "Key implementation choices: face images are always converted to grayscale and "
    "histogram-equalized before recognition/emotion inference to reduce lighting sensitivity; "
    "LBPH confidence is a distance score, so predictions above "
    "<font face='Courier'>RECOGNITION_CONFIDENCE_THRESHOLD</font> are relabeled 'Unknown' rather than a wrong name; "
    "attendance writes are guarded by a database UNIQUE constraint rather than application-level "
    "checks, so a race between two near-simultaneous recognitions cannot double-log a person."
)

# ============================================================ 9. (kept for numbering continuity - merged)
# ============================================================ 9. Screenshots / Results
story.append(PageBreak())
h1("10. Screenshots / Results")
body("Representative CLI sessions demonstrating the system's usability and error handling:")
figure(os.path.join(DOCS, "screenshot_cli_help.png"), "Figure 7: CLI help output listing all subcommands")
figure(os.path.join(DOCS, "screenshot_list_users.png"), "Figure 8: Listing enrolled users")
figure(os.path.join(DOCS, "screenshot_report_summary.png"), "Figure 9: Per-person attendance/emotion summary")
figure(os.path.join(DOCS, "screenshot_error_handling.png"), "Figure 10: Graceful error handling when no face is detected during enrollment")

story.append(PageBreak())
body("Generated analytics charts from a seeded 7-day demo dataset (see scripts/seed_demo_data.py):")
figure(os.path.join(REPORTS, "attendance_trend.png"), "Figure 11: Daily attendance trend")
figure(os.path.join(REPORTS, "emotion_distribution.png"), "Figure 12: Emotion distribution at check-in")

# ============================================================ 10. Testing Approach
story.append(PageBreak())
h1("11. Testing Approach")
body(
    "The project uses <font face='Courier'>pytest</font> with a fully synthetic, seeded data generator "
    "(<font face='Courier'>tests/conftest.py</font>) so the entire suite runs without a webcam or any external "
    "dataset -- important for reproducible grading. Coverage includes:"
)
bullets(
    [
        "<b>Database layer:</b> user creation/lookup, duplicate-attendance rejection, date-range filtering.",
        "<b>Face recognizer:</b> trains on two synthetic, class-separable pattern sets and verifies correct "
        "discrimination between them, plus a ValueError when trained on an empty directory.",
        "<b>Emotion classifier:</b> trains on a 3-class synthetic dataset, verifies persistence "
        "(train &rarr; save &rarr; reload &rarr; predict), and rejects single-class training data.",
        "<b>Analytics:</b> CSV export, per-person summary correctness, and chart-file generation, "
        "including the empty-dataset error path.",
        "<b>Enrollment:</b> input validation (empty name / no images), no-face-detected abort, and a "
        "mocked-detector happy path verifying file save + DB registration logic in isolation.",
    ]
)
body(
    "All persistence-touching tests run against a temporary directory via a <font face='Courier'>tmp_config</font> "
    "pytest fixture, so running the suite never touches real enrollment data or the production database."
)

# ============================================================ 11. Challenges Faced
h1("12. Challenges Faced")
bullets(
    [
        "<b>Choosing lightweight models without sacrificing correctness:</b> the initial instinct was to "
        "reach for deep-learning face embeddings and a CNN emotion classifier, but that would have made "
        "the 'runs from a single pip install, no GPU required' requirement much harder to satisfy. "
        "Settling on LBPH and HOG+SVM required re-validating that they were still faithful to the "
        "course's classical-CV concepts rather than a step down in rigor.",
        "<b>Avoiding double attendance marks:</b> an application-level 'check then insert' approach is "
        "vulnerable to race conditions; moving the uniqueness rule into the database schema itself "
        "(a UNIQUE constraint) made the guarantee robust regardless of calling code.",
        "<b>Testing computer-vision code without a webcam or real face dataset:</b> solved by building a "
        "deterministic synthetic pattern-image generator that still exercises the real training/prediction "
        "code paths for both LBPH and HOG+SVM.",
        "<b>Keeping emotion tagging from ever breaking attendance:</b> emotion prediction is wrapped so "
        "that any failure (e.g. a missing/corrupt model file) logs a warning and continues, rather than "
        "aborting the more important attendance log.",
    ]
)

# ============================================================ 12. Learnings & Key Takeaways
story.append(PageBreak())
h1("13. Learnings & Key Takeaways")
bullets(
    [
        "Classical Computer Vision techniques (Haar cascades, LBPH, HOG+SVM) remain highly practical "
        "for small, controlled-deployment problems, and are far cheaper to set up and run than deep "
        "learning alternatives.",
        "Pushing data-integrity guarantees (like 'one attendance record per person per day') down into "
        "the database schema is more robust than enforcing them only in application code.",
        "A clean module boundary between detection, recognition, and classification (rather than one "
        "monolithic 'vision.py') made it straightforward to unit-test each piece independently and to "
        "reason about where a future upgrade (e.g. a CNN) would plug in.",
        "Designing for CLI-only, headless execution from the start (rather than retrofitting it later) "
        "meaningfully shaped the architecture -- e.g. every webcam-based function has a static-image "
        "equivalent, which is also what made automated testing possible at all.",
    ]
)

# ============================================================ 13. Future Enhancements
h1("14. Future Enhancements")
bullets(
    [
        "Replace LBPH with a deep face-embedding model (e.g. ArcFace) behind the same FaceRecognizer "
        "interface, for better robustness to pose and lighting variation.",
        "Replace HOG+SVM with a CNN trained on the full FER2013 dataset for higher emotion-classification accuracy.",
        "Add a <font face='Courier'>--train-test-split</font> evaluation flag reporting accuracy and a confusion matrix for both models.",
        "Add basic liveness/anti-spoofing detection to prevent a printed photo from being recognized.",
        "Wrap the existing modules in a small REST API for integration with a web-based dashboard, "
        "without changing any core logic.",
    ]
)

# ============================================================ 14. Conclusion  (kept, not in the mandated list but useful)
# Per the mandated list, section 14 is Future Enhancements (done above) and 15 is References.
# ============================================================ 15. References
h1("15. References")
bullets(
    [
        "Ahonen, T., Hadid, A., &amp; Pietik&auml;inen, M. (2006). Face description with local binary "
        "patterns: Application to face recognition. <i>IEEE TPAMI</i>.",
        "Dalal, N., &amp; Triggs, B. (2005). Histograms of oriented gradients for human detection. <i>CVPR</i>.",
        "Viola, P., &amp; Jones, M. (2001). Rapid object detection using a boosted cascade of simple "
        "features. <i>CVPR</i>.",
        "Goodfellow, I. et al. (2013). Challenges in representation learning: A report on three machine "
        "learning contests (origin of the FER2013 dataset).",
        "OpenCV documentation: https://docs.opencv.org/",
        "scikit-learn documentation: https://scikit-learn.org/",
        "scikit-image HOG documentation: https://scikit-image.org/docs/stable/api/skimage.feature.html",
    ]
)

doc = SimpleDocTemplate(
    OUT_PATH,
    pagesize=A4,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    title="Smart Attendance & Wellbeing System - Project Report",
)
doc.build(story)
print(f"Report written to {OUT_PATH}")
