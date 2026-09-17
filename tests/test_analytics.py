import os

import pytest

from src.analytics import Analytics, AnalyticsError
from src.db import Database


def _seed_db(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    alice = db.add_user("Alice")
    bob = db.add_user("Bob")
    db.mark_attendance(alice, "2026-09-01", "09:00:00", 35.0, "happy")
    db.mark_attendance(bob, "2026-09-01", "09:05:00", 40.0, "neutral")
    db.mark_attendance(alice, "2026-09-02", "09:02:00", 33.0, "happy")
    return db


def test_export_csv(tmp_config):
    db = _seed_db(tmp_config)
    analytics = Analytics(db)
    out_path = str(tmp_config / "reports" / "out.csv")
    n = analytics.export_csv(out_path)
    assert n == 3
    assert os.path.exists(out_path)


def test_export_csv_raises_when_empty(tmp_config):
    db = Database(db_path=str(tmp_config / "attendance.db"))
    analytics = Analytics(db)
    with pytest.raises(AnalyticsError):
        analytics.export_csv(str(tmp_config / "out.csv"))


def test_summary_counts_and_dominant_emotion(tmp_config):
    db = _seed_db(tmp_config)
    analytics = Analytics(db)
    summary = analytics.summary()
    alice_row = summary[summary["name"] == "Alice"].iloc[0]
    bob_row = summary[summary["name"] == "Bob"].iloc[0]
    assert alice_row["days_present"] == 2
    assert alice_row["dominant_emotion"] == "happy"
    assert bob_row["days_present"] == 1


def test_plot_attendance_trend_creates_file(tmp_config):
    db = _seed_db(tmp_config)
    analytics = Analytics(db)
    out_path = str(tmp_config / "reports" / "trend.png")
    analytics.plot_attendance_trend(out_path)
    assert os.path.exists(out_path)


def test_plot_emotion_distribution_creates_file(tmp_config):
    db = _seed_db(tmp_config)
    analytics = Analytics(db)
    out_path = str(tmp_config / "reports" / "emotion.png")
    analytics.plot_emotion_distribution(out_path)
    assert os.path.exists(out_path)
