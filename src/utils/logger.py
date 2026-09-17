"""
logger.py
Provides a single shared logger instance used across all modules.
Logs to both console and a rotating file (app.log) for auditability
(non-functional requirement: logging / monitoring).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str = "smart_attendance") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated imports
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        file_handler = RotatingFileHandler(
            config.LOG_PATH, maxBytes=1_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # If the filesystem is read-only or unavailable, fall back to console-only.
        pass

    return logger
