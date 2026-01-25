"""
Prediction module
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from typing import Tuple
import os

def preprocess_single_image(image_path: str or Image.Image,
                           target_size: Tuple[int, int] = (224, 224),
                           grayscale: bool = True,
                           rgb: bool = False) -> np.ndarray:
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image or PIL Image object
        target_size: Target size for resizing
        grayscale: Whether to convert to grayscale
        
    Returns:
        Preprocessed image array
    """
    # Load image
    if isinstance(image_path, str):
        if image_path.endswith('.dcm'):
            import pydicom
            dicom_file = pydicom.dcmread(image_path)
            image = dicom_file.pixel_array
            # Normalize to 0-255
            if image.max() > 255:
                image = ((image - image.min()) / 
                        (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = np.array(Image.open(image_path))
    else:
        image = np.array(image_path)
    
    # Convert to grayscale if needed
    if grayscale and len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = image[:, :, 0]
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension if grayscale
    if grayscale and len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    # Convert to RGB if needed (for transfer learning models)
    if rgb and len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_pneumonia(model: tf.keras.Model,
                     image_path: str or Image.Image,
                     class_names: list = None) -> Tuple[str, float, dict]:
    """
    Predict pneumonia from an image
    
    Args:
        model: Trained Keras model
        image_path: Path to image or PIL Image object
        class_names: List of class names
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Preprocess image
    processed_image = preprocess_single_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    probabilities = predictions[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_class_idx])
    
    # Get class name
    if class_names:
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = f"Class {predicted_class_idx}"
    
    # Create probability dictionary
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        class_name = class_names[i] if class_names else f"Class {i}"
        prob_dict[class_name] = float(prob)
    
    return predicted_class, confidence, prob_dict
