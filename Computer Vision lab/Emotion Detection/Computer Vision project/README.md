# Real-Time Emotion Detection System

A real-time computer vision application that detects human faces from a live
webcam feed and classifies facial expressions into one of seven emotions:
**Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral.**

Built with **Python, OpenCV, and DeepFace** (a pre-trained deep CNN for facial
emotion recognition), the system overlays the detected emotion, a live
confidence breakdown, and performance metrics directly onto the video feed.

---

## Features

- **Real-time face detection** using OpenCV Haar Cascades
- **Deep learning–based emotion classification** (DeepFace CNN, trained on
  the FER2013 dataset under the hood)
- **Temporal smoothing** — predictions are averaged over a rolling window so
  the displayed label doesn't flicker frame-to-frame
- **Frame-skipping optimization** — the expensive CNN inference only runs
  every N frames, keeping the app responsive on CPU-only machines
- **Live confidence distribution bars** for all 7 emotion classes
- **FPS counter and on-screen HUD**
- **Snapshot capture** (`S` key) and **session CSV logging** of the full
  emotion timeline for later analysis (e.g. customer feedback review,
  classroom engagement tracking)
- **Clean modular architecture** — detection, classification, rendering, and
  logging are fully decoupled

---

## Project Structure

```
emotion-detection-system/
├── src/
│   ├── config.py              # All tunable settings in one place
│   ├── face_detector.py       # Haar Cascade face localization
│   ├── emotion_classifier.py  # DeepFace wrapper + smoothing logic
│   ├── ui_overlay.py          # On-screen rendering (boxes, bars, HUD)
│   ├── session_logger.py      # CSV timeline logging + app logging
│   └── main.py                # Application entry point / main loop
├── logs/                      # Auto-generated runtime logs
├── outputs/                   # Snapshots + emotion CSV logs
├── requirements.txt
└── README.md
```

---

## How It Works (Pipeline)

```
Webcam Frame
     │
     ▼
[1] Face Detection (Haar Cascade, grayscale + histogram equalization)
     │   → bounding boxes (x, y, w, h)
     ▼
[2] Crop face region(s)
     │
     ▼
[3] Emotion Inference (DeepFace CNN)
     │   → runs every Nth frame for performance
     │   → smoothed over a rolling window per face
     ▼
[4] Render: bounding box + label + confidence bars + FPS HUD
     │
     ▼
[5] Log emotion change events to CSV
```

This "detect cheaply, classify expensively, only when needed" pattern is the
standard design for real-time CV systems — it's what lets the app run
smoothly even on a laptop CPU with no dedicated GPU.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `deepface` and `tensorflow` are larger packages — first install
> may take a few minutes. On first run, DeepFace will also auto-download its
> pre-trained emotion model weights (~a few MB), so make sure you have
> internet access the first time you run the app.

### 2. Run the application

From the project root:

```bash
python -m src.main
```

### 3. Controls

| Key | Action                    |
|-----|----------------------------|
| `Q` | Quit the application       |
| `S` | Save a snapshot to `outputs/` |

---

## Output

- **`logs/session_<timestamp>.log`** — full runtime log (info, warnings, errors)
- **`outputs/emotion_log_<timestamp>.csv`** — timestamped emotion timeline:

  | timestamp           | face_slot | emotion | confidence |
  |----------------------|-----------|---------|------------|
  | 2026-06-19T14:02:11 | 0         | happy   | 91.42      |
  | 2026-06-19T14:02:14 | 0         | neutral | 76.10      |

- **`outputs/snapshot_<timestamp>.png`** — saved frames with overlays

---

## Configuration

All key parameters can be tuned in `src/config.py` without touching any
logic code, including:

- Camera resolution / index
- Face detection sensitivity (`FACE_MIN_NEIGHBORS`, `FACE_SCALE_FACTOR`)
- Inference frequency (`INFERENCE_EVERY_N_FRAMES`)
- Smoothing window size (`SMOOTHING_WINDOW`)
- UI colors per emotion

---

## Tech Stack

| Component             | Technology                          |
|------------------------|--------------------------------------|
| Face Detection         | OpenCV Haar Cascade Classifier      |
| Emotion Classification | DeepFace (CNN pre-trained on FER2013) |
| Video Capture / UI     | OpenCV                              |
| Logging                | Python `logging` + CSV              |

---

## Possible Extensions (Future Work)

- Swap Haar Cascade for a DNN-based face detector (e.g. MTCNN, RetinaFace)
  for better accuracy under occlusion/profile angles
- Add face tracking (e.g. via centroid tracking or DeepSORT) to maintain
  consistent identity across frames for multi-person scenes
- Train a custom lightweight CNN on FER2013 for offline/edge deployment
  without the DeepFace dependency
- Build a dashboard (e.g. Streamlit) to visualize the emotion CSV logs
  over time — directly applicable to the "customer feedback analysis"
  and "classroom engagement" use cases mentioned in the project brief

---

## Use Cases

- **Education** — gauge student engagement/attention during lectures
- **Healthcare** — supplementary emotional-state monitoring for patients
- **Customer feedback** — passive sentiment reading during product demos
- **Human-Computer Interaction** — adaptive interfaces that respond to
  user mood
