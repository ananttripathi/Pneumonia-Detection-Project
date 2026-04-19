"""
Streamlit app for Pneumonia Detection — loads model from HF Hub at startup.
"""
import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.model import build_model

MODEL_REPO = "ananttripathiak/pneumonia-detection-model"
MODEL_FILENAME = "best_model.weights.h5"

CLASS_LABELS = {0: "Normal", 1: "Lung Opacity", 2: "No Lung Opacity / Not Normal"}

st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
    <style>
    .main-header { font-size:2.2rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:1.5rem; }
    .pred-box { padding:1.2rem; border-radius:10px; background:#f0f2f6; margin:1rem 0; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model from Hugging Face Hub…")
def load_model():
    try:
        weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir="/tmp")
        model = build_model()
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


def preprocess(image: Image.Image) -> np.ndarray:
    import cv2
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_dicom(dcm_path: str) -> np.ndarray:
    import cv2, pydicom
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)
    if img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)


def main():
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection from Chest X-Ray</h1>', unsafe_allow_html=True)
    st.markdown(
        "Upload a chest X-ray image (DICOM or PNG/JPG) to get an AI-assisted classification.\n\n"
        "**⚠️ Disclaimer:** For research and educational purposes only — not a medical diagnostic tool."
    )

    model = load_model()
    if model is None:
        st.warning("Model not available yet. Please train and upload the model first using `run_training.py`.")
        st.stop()

    st.sidebar.header("Classes")
    st.sidebar.markdown("- **Normal** — No pneumonia\n- **Lung Opacity** — Pneumonia detected\n- **No Lung Opacity / Not Normal** — Abnormality, not pneumonia")
    st.sidebar.header("Instructions")
    st.sidebar.markdown("1. Upload a chest X-ray\n2. Wait for analysis\n3. Review prediction and confidence")

    uploaded = st.file_uploader("Upload Chest X-Ray", type=["dcm", "png", "jpg", "jpeg", "tiff", "bmp"])

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            try:
                if uploaded.name.lower().endswith(".dcm"):
                    import tempfile, pydicom, cv2
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    dcm = pydicom.dcmread(tmp_path)
                    arr = dcm.pixel_array
                    if arr.max() > 255:
                        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
                    st.image(arr, use_container_width=True, caption="Chest X-Ray (DICOM)")
                    img_input = preprocess_dicom(tmp_path)
                else:
                    image = Image.open(uploaded)
                    st.image(image, use_container_width=True, caption="Chest X-Ray")
                    img_input = preprocess(image)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.stop()

        with col2:
            st.subheader("Prediction")
            with st.spinner("Analysing…"):
                preds = model.predict(img_input, verbose=0)[0]

            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_LABELS[pred_idx]
            confidence = float(preds[pred_idx])

            st.markdown(f'<div class="pred-box"><h3>Predicted: <b>{pred_class}</b></h3></div>', unsafe_allow_html=True)

            conf_color = "green" if confidence >= 0.7 else ("orange" if confidence >= 0.5 else "red")
            st.markdown(f"**Confidence:** :{conf_color}[{confidence*100:.1f}%]")

            st.subheader("Class Probabilities")
            for idx, label in CLASS_LABELS.items():
                st.progress(float(preds[idx]), text=f"{label}: {preds[idx]*100:.1f}%")

            st.subheader("Recommendation")
            if pred_class == "Lung Opacity":
                st.warning("⚠️ Pneumonia detected. Please consult a healthcare professional.")
            elif pred_class == "No Lung Opacity / Not Normal":
                st.info("ℹ️ Abnormality detected. Further medical evaluation recommended.")
            else:
                st.success("✅ No pneumonia detected. Image appears normal.")
    else:
        st.info("👆 Upload a chest X-ray to get started.")
        with st.expander("How to use"):
            st.markdown("1. Prepare a chest X-ray in DICOM or standard image format\n"
                        "2. Click upload and select the file\n"
                        "3. Wait a few seconds for the model to analyse\n"
                        "4. Review results — always consult a doctor for medical decisions")


if __name__ == "__main__":
    main()
