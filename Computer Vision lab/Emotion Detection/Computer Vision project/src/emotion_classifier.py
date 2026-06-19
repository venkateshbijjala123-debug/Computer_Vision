"""
emotion_classifier.py
----------------------
Emotion inference using DeepFace's pre-trained CNN, with two production-style
additions that the raw library doesn't give you for free:

    1. Frame-skipping: deep inference only every N frames (config-driven),
       since a CNN forward pass is the most expensive step in the pipeline.
    2. Temporal smoothing: predictions are averaged over a sliding window
       per tracked face, so the displayed label doesn't flicker between
       "happy" and "neutral" every frame due to small probability jitter.

This mirrors how emotion recognition is actually deployed in products --
raw per-frame model output is rarely shown directly to the end user.
"""

import logging
from collections import deque, defaultdict

import numpy as np
from deepface import DeepFace

from . import config

logger = logging.getLogger(__name__)


class EmotionResult:
    """Simple container for a single emotion prediction."""

    __slots__ = ("label", "confidence", "distribution")

    def __init__(self, label, confidence, distribution):
        self.label = label                  # e.g. "happy"
        self.confidence = confidence        # 0-100
        self.distribution = distribution    # dict of all class scores


class EmotionClassifier:
    """
    Stateful emotion classifier. One instance should persist across the
    whole session so its smoothing buffers retain temporal context.
    """

    def __init__(self):
        # Each tracked face slot gets its own rolling history buffer.
        self._history = defaultdict(
            lambda: deque(maxlen=config.SMOOTHING_WINDOW)
        )
        self._frame_counter = 0
        self._last_result_by_slot = {}

        logger.info(
            "EmotionClassifier initialized (backend=%s, infer_every=%d frames)",
            config.DEEPFACE_DETECTOR_BACKEND,
            config.INFERENCE_EVERY_N_FRAMES,
        )

    def should_run_inference(self):
        """Frame-skip gate -- call once per frame before classify()."""
        self._frame_counter += 1
        return (self._frame_counter % config.INFERENCE_EVERY_N_FRAMES) == 0

    def classify(self, face_crop_bgr, slot_id=0):
        """
        Run (or reuse) emotion inference for a single face crop.

        Args:
            face_crop_bgr: cropped face region, BGR.
            slot_id: identifier for which tracked face this is (supports
                     multi-face smoothing without mixing histories).

        Returns:
            EmotionResult, or None if no usable prediction is available yet.
        """
        if not self.should_run_inference():
            return self._last_result_by_slot.get(slot_id)

        try:
            analysis = DeepFace.analyze(
                face_crop_bgr,
                actions=["emotion"],
                detector_backend=config.DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=config.DEEPFACE_ENFORCE_DETECTION,
                silent=True,
            )
        except Exception as exc:
            logger.warning("DeepFace inference failed for slot %s: %s", slot_id, exc)
            return self._last_result_by_slot.get(slot_id)

        # DeepFace returns a list (one entry per detected face in the crop)
        if isinstance(analysis, list):
            analysis = analysis[0]

        distribution = analysis.get("emotion", {})
        if not distribution:
            return self._last_result_by_slot.get(slot_id)

        # Push this frame's distribution into the smoothing buffer
        history = self._history[slot_id]
        history.append(distribution)

        smoothed = self._smooth(history)
        label = max(smoothed, key=smoothed.get)
        confidence = smoothed[label]

        result = EmotionResult(label=label, confidence=confidence, distribution=smoothed)
        self._last_result_by_slot[slot_id] = result
        return result

    @staticmethod
    def _smooth(history):
        """Average emotion score distributions across the rolling window."""
        keys = config.EMOTION_LABELS
        stacked = np.array([[d.get(k, 0.0) for k in keys] for d in history])
        mean_scores = stacked.mean(axis=0)
        return dict(zip(keys, mean_scores))

    def reset_slot(self, slot_id):
        """Clear smoothing history for a slot (e.g. when a face is lost)."""
        self._history.pop(slot_id, None)
        self._last_result_by_slot.pop(slot_id, None)
