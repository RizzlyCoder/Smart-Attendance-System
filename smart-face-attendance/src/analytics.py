"""
analytics.py
Functional Module 3: Reporting & Analytics.
Turns raw attendance/emotion logs into CSV exports and charts so the data
collected by modules 1-2 is actually useful to an end user.
"""

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, required for CLI/server execution
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.db import Database
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLUMNS = ["record_id", "name", "date", "time", "confidence", "emotion"]


class AnalyticsError(Exception):
    pass


class Analytics:
    def __init__(self, db: Database = None):
        self.db = db or Database()

    def _dataframe(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        rows = self.db.get_attendance(start_date, end_date)
        return pd.DataFrame(rows, columns=COLUMNS)

    def export_csv(
        self, out_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> int:
        df = self._dataframe(start_date, end_date)
        if df.empty:
            raise AnalyticsError("No attendance records found for the given date range.")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Exported %d attendance record(s) to %s", len(df), out_path)
        return len(df)

    def summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Per-user attendance count and dominant emotion."""
        df = self._dataframe(start_date, end_date)
        if df.empty:
            raise AnalyticsError("No attendance records found for the given date range.")
        counts = df.groupby("name").size().rename("days_present")
        dominant_emotion = (
            df.dropna(subset=["emotion"])
            .groupby("name")["emotion"]
            .agg(lambda s: s.value_counts().idxmax() if not s.empty else None)
            .rename("dominant_emotion")
        )
        result = pd.concat([counts, dominant_emotion], axis=1).reset_index()
        return result

    def plot_attendance_trend(self, out_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        df = self._dataframe(start_date, end_date)
        if df.empty:
            raise AnalyticsError("No attendance records found for the given date range.")
        daily_counts = df.groupby("date").size()
        plt.figure(figsize=(8, 4))
        daily_counts.plot(kind="bar", color="#3B82F6")
        plt.title("Daily Attendance Count")
        plt.xlabel("Date")
        plt.ylabel("People Present")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved attendance trend chart to %s", out_path)

    def plot_emotion_distribution(self, out_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        df = self._dataframe(start_date, end_date).dropna(subset=["emotion"])
        if df.empty:
            raise AnalyticsError("No emotion-tagged records found for the given date range.")
        emotion_counts = df["emotion"].value_counts()
        plt.figure(figsize=(6, 6))
        emotion_counts.plot(kind="pie", autopct="%1.0f%%")
        plt.title("Emotion Distribution at Check-in")
        plt.ylabel("")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved emotion distribution chart to %s", out_path)
