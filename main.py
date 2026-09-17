#!/usr/bin/env python3
"""
main.py
Command-line entry point for the Smart Attendance & Wellbeing System.
Every capability of the project is reachable from the terminal, satisfying
the "must be fully executable via CLI" submission requirement.

Run `python main.py -h` for full usage, or see README.md for a walkthrough.
"""

import argparse
import glob
import sys

from src.analytics import Analytics, AnalyticsError
from src.attendance import AttendanceSession
from src.db import Database
from src.emotion_model import EmotionClassifier
from src.enrollment import Enrollment, EnrollmentError
from src.face_recognizer import FaceRecognizer
from src.utils.logger import get_logger

logger = get_logger("cli")


def cmd_enroll(args):
    enrollment = Enrollment()
    try:
        if args.images:
            paths = []
            for pattern in args.images:
                paths.extend(sorted(glob.glob(pattern)))
            saved = enrollment.enroll_from_images(args.name, paths)
        else:
            saved = enrollment.enroll_from_webcam(args.name, num_samples=args.samples)
        print(f"[OK] Enrolled '{args.name}' with {saved} face sample(s).")
    except EnrollmentError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_train_recognizer(args):
    recognizer = FaceRecognizer()
    try:
        n = recognizer.train_from_directory()
        print(f"[OK] Trained face recognizer on {n} image(s). Model saved to models/lbph_recognizer.yml")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_train_emotion(args):
    clf = EmotionClassifier()
    try:
        n, labels = clf.train_from_directory(args.dataset)
        print(f"[OK] Trained emotion classifier on {n} image(s) across classes: {labels}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_attendance(args):
    session = AttendanceSession(use_emotion=not args.no_emotion)
    if args.image:
        results = session.run_on_image(args.image)
        if not results:
            print("[INFO] No face detected in the image.")
        for name, confidence, emotion, marked in results:
            status = "marked" if marked else "already marked / unrecognized"
            print(f"  {name:<20} confidence={confidence:6.1f}  emotion={emotion or '-':<10} [{status}]")
    else:
        print("[INFO] Starting webcam session. Press 'q' in the video window to stop.")
        session.run_on_webcam(camera_index=args.camera)


def cmd_report(args):
    analytics = Analytics()
    try:
        if args.csv:
            analytics.export_csv(args.csv, args.start, args.end)
            print(f"[OK] CSV exported to {args.csv}")
        if args.trend_chart:
            analytics.plot_attendance_trend(args.trend_chart, args.start, args.end)
            print(f"[OK] Attendance trend chart saved to {args.trend_chart}")
        if args.emotion_chart:
            analytics.plot_emotion_distribution(args.emotion_chart, args.start, args.end)
            print(f"[OK] Emotion distribution chart saved to {args.emotion_chart}")
        if args.summary:
            df = analytics.summary(args.start, args.end)
            print(df.to_string(index=False))
        if not any([args.csv, args.trend_chart, args.emotion_chart, args.summary]):
            print("[INFO] Nothing to do -- pass --csv, --trend-chart, --emotion-chart, and/or --summary.")
    except AnalyticsError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def cmd_list_users(args):
    db = Database()
    users = db.list_users()
    if not users:
        print("[INFO] No users enrolled yet.")
        return
    for user_id, name in users:
        print(f"  #{user_id:<4} {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-attendance",
        description="Smart Attendance & Wellbeing System -- face recognition + emotion analytics from the CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Register a new person (Module 1)")
    p_enroll.add_argument("--name", required=True, help="Full name of the person to enroll")
    p_enroll.add_argument(
        "--images", nargs="+", help="Glob pattern(s) of face image files, e.g. data/raw/alice/*.jpg"
    )
    p_enroll.add_argument("--samples", type=int, default=20, help="Number of webcam samples to capture (default 20)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_train_rec = sub.add_parser("train-recognizer", help="Train the LBPH face recognizer on enrolled faces")
    p_train_rec.set_defaults(func=cmd_train_recognizer)

    p_train_emo = sub.add_parser("train-emotion", help="Train the HOG+SVM emotion classifier")
    p_train_emo.add_argument(
        "--dataset", required=True, help="Path to a folder with one subfolder per emotion label"
    )
    p_train_emo.set_defaults(func=cmd_train_emotion)

    p_att = sub.add_parser("attendance", help="Run recognition + mark attendance (Module 2)")
    p_att.add_argument("--image", help="Path to a static image instead of using the webcam")
    p_att.add_argument("--camera", type=int, default=0, help="Camera index for webcam mode (default 0)")
    p_att.add_argument("--no-emotion", action="store_true", help="Disable emotion tagging")
    p_att.set_defaults(func=cmd_attendance)

    p_report = sub.add_parser("report", help="Generate analytics/reports (Module 3)")
    p_report.add_argument("--csv", help="Path to write a CSV export of attendance records")
    p_report.add_argument("--trend-chart", help="Path to save a daily attendance bar chart (PNG)")
    p_report.add_argument("--emotion-chart", help="Path to save an emotion distribution pie chart (PNG)")
    p_report.add_argument("--summary", action="store_true", help="Print a per-person summary table")
    p_report.add_argument("--start", help="Start date filter YYYY-MM-DD")
    p_report.add_argument("--end", help="End date filter YYYY-MM-DD")
    p_report.set_defaults(func=cmd_report)

    p_users = sub.add_parser("list-users", help="List all enrolled users")
    p_users.set_defaults(func=cmd_list_users)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
