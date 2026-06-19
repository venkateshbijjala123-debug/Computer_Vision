"""
face_detector.py
-----------------
Lightweight face localization using OpenCV's Haar Cascade classifier.

Design rationale:
    Running a deep emotion-recognition model on the *entire* frame for
    every frame is wasteful. Instead, this module performs cheap,
    fast face localization first. Only the cropped face region(s) are
    then handed off to the (much heavier) emotion classifier. This
    two-stage "detect, then classify" pipeline is the standard
    approach in production-grade real-time vision systems.
"""

import cv2
import logging
from . import config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps a Haar Cascade face detector with sane defaults."""

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades + config.HAAR_CASCADE_PATH
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)

        if self._cascade.empty():
            raise IOError(
                f"Failed to load Haar Cascade from '{cascade_path}'. "
                "Verify your OpenCV installation includes the data files."
            )

        logger.info("FaceDetector initialized using cascade: %s", cascade_path)

    def detect(self, frame_bgr):
        """
        Detect faces in a BGR frame.

        Args:
            frame_bgr: np.ndarray, raw BGR frame from the camera.

        Returns:
            List of (x, y, w, h) bounding boxes, sorted by area (largest first)
            so downstream code can easily prioritize the most prominent face.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves robustness under poor lighting

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        return faces
