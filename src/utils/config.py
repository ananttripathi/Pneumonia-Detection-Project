"""
Configuration file for the Pneumonia Detection Project
"""
import os

# Data paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs', 'training_logs')

# Image parameters
IMG_SIZE = (224, 224)
IMG_CHANNELS = 1  # Grayscale
NUM_CLASSES = 3

# Class labels
CLASS_LABELS = {
    0: 'Normal',
    1: 'Lung Opacity',
    2: 'No Lung Opacity / Not Normal'
}

CLASS_LABELS_REVERSE = {
    'Normal': 0,
    'Lung Opacity': 1,
    'No Lung Opacity / Not Normal': 2
}

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
