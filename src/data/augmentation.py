"""
Data augmentation module for training
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmentation_generator():
    """
    Create ImageDataGenerator with augmentation parameters
    
    Returns:
        ImageDataGenerator with augmentation settings
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    return datagen

def create_train_generator(X_train, y_train, batch_size=32, augment=True):
    """
    Create training data generator with optional augmentation
    
    Args:
        X_train: Training images
        y_train: Training labels
        batch_size: Batch size
        augment: Whether to apply augmentation
        
    Returns:
        Data generator
    """
    if augment:
        datagen = create_augmentation_generator()
        generator = datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
    else:
        # Simple generator without augmentation
        datagen = ImageDataGenerator()
        generator = datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
    
    return generator
