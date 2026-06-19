"""
ui_overlay.py
-------------
All on-screen rendering logic lives here, separated from detection/inference
logic. This separation of concerns (CV pipeline vs. presentation layer) is
what keeps the codebase maintainable as features grow.
"""

import time
import cv2
from . import config


class FPSTracker:
    """Simple exponential-moving-average FPS counter."""

    def __init__(self, alpha=0.1):
        self._alpha = alpha
        self._fps = 0.0
        self._last_time = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = (
                instant_fps if self._fps == 0
                else (self._alpha * instant_fps + (1 - self._alpha) * self._fps)
            )
        return self._fps


def draw_face_box(frame, box, label, confidence):
    """Draw bounding box + emotion label above a detected face."""
    x, y, w, h = box
    color = config.EMOTION_COLORS.get(label, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BOX_THICKNESS)

    text = f"{label.upper()} {confidence:.0f}%" if label else "Detecting..."
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # filled label background for legibility over any video content
    cv2.rectangle(
        frame,
        (x, y - text_h - 14),
        (x + text_w + 10, y),
        color,
        -1,
    )
    cv2.putText(
        frame, text, (x + 5, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_confidence_bars(frame, distribution, origin=(15, 60)):
    """Draw a small live bar chart of all emotion class scores (top-left)."""
    if not distribution:
        return

    x0, y0 = origin
    bar_width = 150
    bar_height = 14
    spacing = 20

    cv2.rectangle(
        frame,
        (x0 - 8, y0 - 25),
        (x0 + bar_width + 60, y0 + spacing * len(distribution) + 5),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame, "Emotion Distribution", (x0 - 2, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

    for i, label in enumerate(config.EMOTION_LABELS):
        score = distribution.get(label, 0.0)
        y = y0 + i * spacing
        color = config.EMOTION_COLORS.get(label, (200, 200, 200))

        filled = int((score / 100.0) * bar_width)
        cv2.rectangle(frame, (x0, y), (x0 + bar_width, y + bar_height), (60, 60, 60), 1)
        cv2.rectangle(frame, (x0, y), (x0 + filled, y + bar_height), color, -1)

        cv2.putText(
            frame, f"{label[:4]} {score:4.1f}%", (x0 + bar_width + 8, y + bar_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )


def draw_hud(frame, fps, face_count):
    """Top status bar: FPS, face count, app title."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.putText(
        frame, config.WINDOW_NAME, (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    status = f"FPS: {fps:4.1f}   Faces: {face_count}   [Q] Quit  [S] Save Snapshot"
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame, status, (w - text_size[0] - 10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA
    )
