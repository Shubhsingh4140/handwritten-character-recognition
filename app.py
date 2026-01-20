import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Handwritten Character Recognition", layout="centered")

st.title("✍️ Handwritten Character Recognition")
st.caption("Supports digits (0–9) and uppercase letters (A–Z)")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emnist_model.keras")

model = load_model()

# -----------------------------
# EMNIST Label Map (Balanced)
# -----------------------------
# EMNIST Balanced label mapping (47 classes)
label_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# -----------------------------
# Image Preprocessing (Shared)
# -----------------------------
def preprocess_image(img_gray):
    # Resize
    img = cv2.resize(img_gray, (28, 28))

    # Normalize
    img = img / 255.0

    # Add channel & batch dims
    img = img.reshape(1, 28, 28, 1)

    return img

# -----------------------------
# Prediction Function
# -----------------------------
def predict(img):
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]
    return label_map[idx], confidence

# -----------------------------
# Input Mode Selector
# -----------------------------
mode = st.radio("Choose input method:", ["✍️ Draw", "📁 Upload Image"])

# ======================================================
# ✍️ DRAWING CANVAS (BEST ACCURACY)
# ======================================================
if mode == "✍️ Draw":
    st.subheader("Draw a character")

    canvas = st_canvas(
        fill_color="white",
        stroke_width=12,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)

        # Convert RGBA → Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Invert (EMNIST expects white on black)
        gray = cv2.bitwise_not(gray)

        processed = preprocess_image(gray)
        pred, conf = predict(processed)

        st.success(f"Predicted Character: {pred}")
        st.info(f"Confidence: {conf*100:.2f}%")

        if conf < 0.7:
            st.warning("⚠️ Uncertain prediction. Please draw more clearly.")

# ======================================================
# 📁 IMAGE UPLOAD (LOWER ACCURACY)
# ======================================================
else:
    st.subheader("Upload a handwritten character image")

    uploaded = st.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("L")
        img_np = np.array(image)

        st.image(image, caption="Uploaded Image", width=150)

        # Invert if background is white
        if np.mean(img_np) > 127:
            img_np = cv2.bitwise_not(img_np)

        processed = preprocess_image(img_np)
        pred, conf = predict(processed)

        st.success(f"Predicted Character: {pred}")
        st.info(f"Confidence: {conf*100:.2f}%")

        if conf < 0.7:
            st.warning("⚠️ Uncertain prediction. Consider using the drawing canvas.")
