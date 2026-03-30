import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load model
model = load_model("mask_detector.h5")

# Voice
engine = pyttsx3.init()
engine.setProperty('rate', 140)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("✅ Camera started")

last_spoken_time = 0
cooldown = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        # ✅ Correct size
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face, verbose=0)[0]

        mask_prob = prediction[0]
        no_mask_prob = prediction[1]

        # 🔥 FIXED DECISION LOGIC
        if mask_prob > no_mask_prob:
            label = "Mask"
            confidence = mask_prob
            color = (0, 255, 0)
        else:
            label = "No Mask"
            confidence = no_mask_prob
            color = (0, 0, 255)

        # 🔊 Voice (more relaxed threshold)
        if current_time - last_spoken_time > cooldown:

            if label == "No Mask" and confidence > 0.60:
                for _ in range(3):
                    engine.say("Please wear a mask")
                engine.runAndWait()
                last_spoken_time = current_time

            elif label == "Mask" and confidence > 0.60:
                engine.say("Perfect, you are wearing the mask")
                engine.runAndWait()
                last_spoken_time = current_time

        text = f"{label} ({confidence:.2f})"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Mask Detection 😷", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()