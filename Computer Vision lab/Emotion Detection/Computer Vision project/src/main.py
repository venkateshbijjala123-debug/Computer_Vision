"""
main.py
-------
Entry point for the Real-Time Emotion Detection System.

Pipeline per frame:
    1. Capture frame from webcam
    2. Detect face bounding boxes (FaceDetector / Haar Cascade)
    3. For each face, run (or reuse) emotion inference (EmotionClassifier / DeepFace)
    4. Render bounding boxes, labels, confidence bars, HUD (ui_overlay)
    5. Optionally log the result to CSV (session_logger)

Run with:
    python -m src.main
"""

import logging
import time

import cv2

from . import config
from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .session_logger import setup_logging, EmotionSessionRecorder
from . import ui_overlay

logger = logging.getLogger(__name__)


class EmotionDetectionApp:
    """Orchestrates capture, detection, classification, and rendering."""

    def __init__(self):
        self.detector = FaceDetector()
        self.classifier = EmotionClassifier()
        self.fps_tracker = ui_overlay.FPSTracker()
        self.recorder = EmotionSessionRecorder() if config.RECORD_SESSION_LOG else None

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {config.CAMERA_INDEX}. "
                "Check that a webcam is connected and not in use by another app."
            )

        self._last_logged_label = {}
        logger.info("Camera opened successfully (index=%d).", config.CAMERA_INDEX)

    def run(self):
        logger.info("Starting main loop. Press 'Q' to quit, 'S' to save a snapshot.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    logger.error("Failed to read frame from camera. Exiting loop.")
                    break

                if config.FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)

                self._process_frame(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed. Shutting down.")
                    break
                elif key == ord("s"):
                    self._save_snapshot(frame)

        finally:
            self.cleanup()

    def _process_frame(self, frame):
        faces = self.detector.detect(frame)
        latest_distribution = None

        for slot_id, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y + h, x:x + w]
            if face_crop.size == 0:
                continue

            result = self.classifier.classify(face_crop, slot_id=slot_id)

            if result is not None:
                ui_overlay.draw_face_box(frame, (x, y, w, h), result.label, result.confidence)
                latest_distribution = result.distribution
                self._maybe_log(slot_id, result)
            else:
                ui_overlay.draw_face_box(frame, (x, y, w, h), None, 0)

        if config.SHOW_CONFIDENCE_BAR and latest_distribution:
            ui_overlay.draw_confidence_bars(frame, latest_distribution)

        fps = self.fps_tracker.tick()
        ui_overlay.draw_hud(frame, fps, len(faces))

        cv2.imshow(config.WINDOW_NAME, frame)

    def _maybe_log(self, slot_id, result):
        """Only write a new CSV row when the label actually changes, to avoid
        flooding the log file with duplicate rows every frame."""
        if not self.recorder:
            return
        if self._last_logged_label.get(slot_id) != result.label:
            self.recorder.log(slot_id, result.label, result.confidence)
            self._last_logged_label[slot_id] = result.label

    def _save_snapshot(self, frame):
        path = f"{config.OUTPUT_DIR}/snapshot_{int(time.time())}.png"
        cv2.imwrite(path, frame)
        logger.info("Snapshot saved to %s", path)

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.recorder:
            self.recorder.close()
            logger.info("Session emotion log saved to %s", self.recorder.path)


def main():
    setup_logging()
    logger.info("=== Real-Time Emotion Detection System: Starting ===")
    app = EmotionDetectionApp()
    app.run()
    logger.info("=== Session ended ===")


if __name__ == "__main__":
    main()
