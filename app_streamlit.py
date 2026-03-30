import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load model
model = load_model("mask_detector.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

st.set_page_config(page_title="Face Mask Detection 😷")

st.title("😷 Face Mask Detection App")
st.write("Upload Image or Use Live Camera")

# ---------------- IMAGE UPLOAD ---------------- #
st.subheader("📸 Upload Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        prediction = model.predict(face)[0]

        if prediction[0] > prediction[1]:
            label = "😷 Mask"
            color = (0, 255, 0)
            st.success("Mask Detected ✅")
        else:
            label = "❌ No Mask"
            color = (0, 0, 255)
            st.error("No Mask Detected ⚠️")

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(img)

# ---------------- CAMERA SECTION ---------------- #
st.subheader("🎥 Live Camera Detection")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            try:
                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                face = np.reshape(face, (1, 224, 224, 3))

                prediction = model.predict(face, verbose=0)[0]

                if prediction[0] > prediction[1]:
                    label = "😷 Mask"
                    color = (0, 255, 0)
                else:
                    label = "❌ No Mask"
                    color = (0, 0, 255)

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except:
                pass

        return img

webrtc_streamer(
    key="mask-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.markdown("Made with ❤️ by Mayuri")