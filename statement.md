# Problem Statement

## Problem Statement

Manual attendance tracking (roll calls, sign-in sheets) is slow, easy to falsify, and gives no insight beyond a bare "present/absent" record. Organizations — classrooms, small offices, workshops — need a lightweight way to (a) confirm who is actually present using something harder to fake than a signature, and (b) get a rough, non-intrusive sense of group wellbeing over time, without adopting a heavyweight commercial HR/biometric platform.

This project builds a **Smart Attendance & Wellbeing System**: a command-line application that enrolls people by face, automatically recognizes and logs their attendance from a photo or webcam feed, and tags each check-in with a lightweight emotion estimate — giving a facilitator both an attendance register and a rough wellbeing trend line, generated entirely from computer-vision techniques taught in this course (face detection, feature-based recognition, and classical machine-learning classification).

## Scope of the Project

**In scope:**
- Enrolling a fixed, known roster of people from photos or webcam captures.
- Detecting faces (Haar cascades) and recognizing enrolled identities (OpenCV LBPH).
- Classifying a coarse emotion label per check-in (HOG features + SVM).
- Logging attendance once per person per day to a local SQLite database.
- Exporting CSV reports and generating attendance/emotion charts.
- Running end-to-end from the command line, including on a headless machine using static image input.

**Out of scope:**
- Large-scale, "in the wild" face recognition (thousands of unknown identities, uncontrolled conditions).
- Liveness/anti-spoofing detection (a printed photo is not defended against).
- A graphical user interface or mobile app — this is explicitly a CLI tool per submission requirements.
- Clinical-grade emotion/mental-health assessment — the emotion tag is a coarse analytics signal, not a diagnostic tool.

## Target Users

- Course instructors or small workshop facilitators tracking attendance for a class roster of a few dozen people.
- Small office/team leads who want a lightweight, self-hosted attendance log without third-party biometric SaaS.
- Students/developers studying classical Computer Vision techniques (Haar cascades, LBPH, HOG+SVM) who want a complete, runnable, end-to-end reference project rather than an isolated notebook.

## High-Level Features

1. **Enrollment Module** — register a new person from a batch of photos or a live webcam session; face crops are normalized (grayscale, histogram-equalized, resized) and stored per person.
2. **Attendance Module** — detects and recognizes faces in an image, video frame, or webcam stream; classifies the detected person's emotion; logs a single attendance record per person per day to SQLite, with graceful handling of unknown faces and missing emotion models.
3. **Analytics & Reporting Module** — exports attendance data to CSV, computes a per-person summary (days present, dominant emotion), and renders an attendance-trend bar chart and an emotion-distribution pie chart.

All three modules are exposed through a single, self-documenting command-line interface (`main.py`), so the entire system can be operated with no GUI.
