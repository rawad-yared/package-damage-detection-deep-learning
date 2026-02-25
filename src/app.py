# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TensorFlow log noise

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Package Damage Detection", page_icon="📦", layout="centered")

st.title("📦 Package Damage Detection")
st.write("Choose a model, then upload an image or take a photo to predict **Damaged vs Intact**.")

IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Custom CNN": os.path.join(BASE_DIR, "models", "custom_cnn_best.keras"),
    "MobileNetV2 (Transfer Learning)": os.path.join(BASE_DIR, "models", "mobilenetv2_stage1_best.keras"),
}

# IMPORTANT: ensure this matches the class order used during training
# (train_ds.class_names printed during training time)
CLASS_NAMES = ["damaged", "intact"]  # 0, 1


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Return array shape (1, 224, 224, 3) float32 in [0..255]."""
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def warmup_model(model):
    """Run one dummy prediction to avoid slow first inference."""
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)


# ---------------------------
# Sidebar controls + model load EARLY
# ---------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select model", list(MODEL_PATHS.keys()))
threshold = st.sidebar.slider("Decision threshold (for class 1)", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("Threshold applies to probability of class 1 (CLASS_NAMES[1]).")

model_path = MODEL_PATHS[model_choice]

try:
    with st.spinner(f"Loading model: {model_choice} ... (first time can be slow)"):
        model = load_model_cached(model_path)

    # Warm-up makes the first real prediction much faster
    with st.spinner("Warming up model..."):
        warmup_model(model)

except Exception as e:
    st.error("Failed to load the selected model. Check the file path and TensorFlow installation.")
    st.write("Model path:", model_path)
    st.exception(e)
    st.stop()


# ---------------------------
# Image input
# ---------------------------
st.subheader("1) Provide an image")
tab1, tab2 = st.tabs(["📤 Upload", "📷 Camera"])

img = None

with tab1:
    uploaded = st.file_uploader("Upload a package image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        img = Image.open(uploaded)

with tab2:
    camera = st.camera_input("Take a photo")
    if camera is not None:
        img = Image.open(camera)

if img is None:
    st.info("Upload an image or take a photo to run prediction.")
    st.stop()

st.image(img, caption="Input image", width=500)


# ---------------------------
# Prediction
# ---------------------------
st.subheader("2) Prediction")

try:
    x = preprocess_image(img)

    with st.spinner("Running inference..."):
        prob_class1 = float(model.predict(x, verbose=0).ravel()[0])  # sigmoid output: P(class 1)

    pred_idx = 1 if prob_class1 >= threshold else 0
    pred_label = CLASS_NAMES[pred_idx]
    confidence = prob_class1 if pred_idx == 1 else (1.0 - prob_class1)

    col1, col2 = st.columns(2)
    col1.metric("Predicted class", pred_label)
    col2.metric("Confidence", f"{confidence*100:.1f}%")

    st.write("### Probability")
    st.write(f"P(**{CLASS_NAMES[1]}**) (class 1): **{prob_class1:.3f}**")
    st.progress(min(max(prob_class1, 0.0), 1.0))

    with st.expander("Details"):
        st.write("Model selected:", model_choice)
        st.write("Model path:", model_path)
        st.write("Threshold:", threshold)
        st.write("Class mapping:", {CLASS_NAMES[0]: 0, CLASS_NAMES[1]: 1})
        st.write("Raw sigmoid output (P(class 1)):", prob_class1)

except Exception as e:
    st.error("The app crashed while running prediction.")
    st.exception(e)
