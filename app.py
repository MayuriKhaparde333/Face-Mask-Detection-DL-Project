import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model("mask_detector.h5")

# UI Design
st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>😷 Face Mask Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload an image to detect mask</h4>", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img_resized)[0]

    mask, no_mask = prediction

    st.markdown("## 📊 Prediction Result")

    # Result Output
    if mask > no_mask:
        st.success("✅ Mask Detected 😷")
    else:
        st.error("❌ No Mask Detected 🚨")

    # Confidence Score
    st.write(f"😷 Mask Confidence: {mask*100:.2f}%")
    st.write(f"🚫 No Mask Confidence: {no_mask*100:.2f}%")

    # Graph
    st.markdown("## 📊 Confidence Graph")
    labels = ["Mask", "No Mask"]
    values = [mask, no_mask]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)