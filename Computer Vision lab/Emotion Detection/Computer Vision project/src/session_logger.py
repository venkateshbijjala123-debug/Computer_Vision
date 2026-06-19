"""
session_logger.py
------------------
Records a timestamped emotion timeline to CSV for the session, and sets up
standard Python logging for diagnostics. Persisting session data is what
turns a one-off demo script into something that supports the "customer
feedback analysis" and "healthcare monitoring" use cases mentioned in the
project brief -- those require a historical record, not just a live view.
"""

import csv
import logging
import os
from datetime import datetime

from . import config


def setup_logging():
    """Configure root logger to write to both console and a log file."""
    log_path = os.path.join(
        config.LOG_DIR, f"session_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


class EmotionSessionRecorder:
    """Appends one row per recognized emotion event to a CSV timeline."""

    def __init__(self):
        self._path = os.path.join(
            config.OUTPUT_DIR, f"emotion_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "face_slot", "emotion", "confidence"])
        self._file.flush()

    def log(self, slot_id, label, confidence):
        self._writer.writerow(
            [datetime.now().isoformat(timespec="seconds"), slot_id, label, f"{confidence:.2f}"]
        )
        self._file.flush()

    @property
    def path(self):
        return self._path

    def close(self):
        if not self._file.closed:
            self._file.close()
