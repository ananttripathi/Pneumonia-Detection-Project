"""
Streamlit deployment app for Pneumonia Detection
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.predict import predict_pneumonia, preprocess_single_image
from src.utils.config import CLASS_LABELS, BEST_MODEL_PATH

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if os.path.exists(BEST_MODEL_PATH):
            model = tf.keras.models.load_model(BEST_MODEL_PATH)
            return model
        else:
            st.error(f"Model not found at {BEST_MODEL_PATH}")
            st.info("Please train a model first using the notebooks.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Pneumonia Detection from Chest X-Ray</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses deep learning to detect pneumonia from chest X-ray images.
    Upload a chest X-ray image (DICOM or standard image format) to get a prediction.
    
    **⚠️ Disclaimer**: This tool is for research and educational purposes only. 
    It should not be used as the sole basis for medical diagnosis.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This application classifies chest X-ray images into three categories:
    - **Normal**: No pneumonia detected
    - **Lung Opacity**: Pneumonia detected
    - **No Lung Opacity / Not Normal**: Abnormality present but not pneumonia
    """)
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image
    2. Wait for processing
    3. View prediction and confidence scores
    """)
    
    # File upload
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['dcm', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            try:
                if uploaded_file.name.endswith('.dcm'):
                    # Handle DICOM
                    import pydicom
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    dicom_file = pydicom.dcmread(tmp_path)
                    image_array = dicom_file.pixel_array
                    
                    # Normalize for display
                    if image_array.max() > 255:
                        image_array = ((image_array - image_array.min()) / 
                                      (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                    
                    st.image(image_array, use_container_width=True, caption="Chest X-Ray Image")
                    image_for_pred = tmp_path
                else:
                    # Handle regular image
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption="Chest X-Ray Image")
                    image_for_pred = image
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction
            try:
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, prob_dict = predict_pneumonia(
                        model, 
                        image_for_pred,
                        class_names=list(CLASS_LABELS.values())
                    )
                
                # Display prediction
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Predicted Class: **{predicted_class}**")
                
                # Confidence indicator
                if confidence >= 0.7:
                    conf_class = "confidence-high"
                elif confidence >= 0.5:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                
                st.markdown(f'<p class="{conf_class}">Confidence: {confidence*100:.2f}%</p>', 
                           unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Probability distribution
                st.subheader("Probability Distribution")
                for class_name, prob in prob_dict.items():
                    st.progress(prob)
                    st.text(f"{class_name}: {prob*100:.2f}%")
                
                # Recommendations
                st.subheader("Recommendations")
                if predicted_class == "Lung Opacity":
                    st.warning("⚠️ Pneumonia detected. Please consult with a healthcare professional for further evaluation and treatment.")
                elif predicted_class == "No Lung Opacity / Not Normal":
                    st.info("ℹ️ Abnormality detected but not pneumonia. Further medical evaluation may be recommended.")
                else:
                    st.success("✅ No pneumonia detected. Image appears normal.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    else:
        # Show sample or instructions
        st.info("👆 Please upload a chest X-ray image to get started.")
        
        # Example section
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Prepare your image**: Ensure the image is a chest X-ray in DICOM (.dcm) or standard image format
            2. **Upload**: Click the upload area above and select your image file
            3. **Wait for analysis**: The model will process the image (usually takes a few seconds)
            4. **Review results**: Check the prediction, confidence scores, and recommendations
            5. **Consult professionals**: Always consult qualified healthcare professionals for medical decisions
            """)

if __name__ == "__main__":
    main()
