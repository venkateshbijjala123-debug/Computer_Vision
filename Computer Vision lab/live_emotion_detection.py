import os
import sys
import urllib.request
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

MODEL_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
MODEL_PATH = "facial_expression_model_weights.h5"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model weights (~50MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")


def build_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))
    return model


def predict_emotion(face_roi, model):
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=-1)
    preds = model.predict(face_roi, verbose=0)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], preds[idx]


def analyze_image(image_path, model, face_cascade):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

    if len(faces) == 0:
        print("No face detected in the image.")
    else:
        print(f"Detected {len(faces)} face(s):")

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        emotion, confidence = predict_emotion(face_roi, model)

        print(f"  -> Emotion: {emotion} ({confidence * 100:.1f}%)")

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y + h), (x + w, y + h + 35), (0, 255, 0), -1)
        cv2.putText(img, f"{emotion} ({confidence * 100:.1f}%)", (x + 5, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        bar_x, bar_y, bar_w = x, y + h + 40, w
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        fill = int(bar_w * confidence)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + 10),
                      (0, 255, 0) if confidence > 0.5 else (0, 255, 255), -1)

    cv2.imshow("Emotion Detection - " + image_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(model, face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        current_emotion = None
        current_conf = 0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion, confidence = predict_emotion(face_roi, model)
            current_emotion = emotion
            current_conf = confidence

            label = f"{emotion} ({confidence * 100:.1f}%)"
            print(f"\rDetected: {label}  ", end="")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_y = y + h + 25
            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 35), (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            bar_x, bar_y, bar_w = x, y + h + 40, w
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
            fill = int(bar_w * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 10),
                          (0, 255, 0) if confidence > 0.5 else (0, 255, 255), -1)

        if current_emotion:
            banner_y = 50
            cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_y + 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Emotion: {current_emotion}", (20, banner_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


def main():
    download_model()

    model = build_model()
    model.load_weights(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if len(sys.argv) > 1:
        analyze_image(sys.argv[1], model, face_cascade)
    else:
        run_webcam(model, face_cascade)


if __name__ == "__main__":
    main()
