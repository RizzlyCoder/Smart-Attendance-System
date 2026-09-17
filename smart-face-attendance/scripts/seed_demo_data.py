#!/usr/bin/env python3
"""
scripts/seed_demo_data.py
Optional convenience script: populates the attendance database with a few
days of sample records so `python main.py report ...` has something to
show immediately, without requiring a webcam or a real photo dataset.
This is a DEMO/GRADING convenience only -- it is not part of the core
attendance pipeline (real usage populates the DB via `attendance`).
"""

import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import Database

NAMES = ["Alice", "Bob", "Charlie", "Diana"]
EMOTIONS = ["happy", "neutral", "sad", "surprise", "angry"]


def main():
    db = Database()
    user_ids = {}
    for name in NAMES:
        existing = db.get_user_by_name(name)
        user_ids[name] = existing[0] if existing else db.add_user(name)

    today = datetime.now().date()
    random.seed(42)
    inserted = 0
    for day_offset in range(7, 0, -1):
        date_str = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for name in NAMES:
            if random.random() < 0.85:  # simulate occasional absence
                hour = random.randint(8, 9)
                minute = random.randint(0, 59)
                time_str = f"{hour:02d}:{minute:02d}:00"
                confidence = round(random.uniform(20, 60), 1)
                emotion = random.choice(EMOTIONS)
                if db.mark_attendance(user_ids[name], date_str, time_str, confidence, emotion):
                    inserted += 1

    print(f"[OK] Seeded {inserted} demo attendance record(s) across {len(NAMES)} users.")


if __name__ == "__main__":
    main()
