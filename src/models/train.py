"""
Model training module
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from typing import Dict, Tuple
import json

def calculate_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )
    
    return dict(zip(classes, class_weights))

def create_callbacks(model_save_path: str, 
                    logs_dir: str,
                    patience: int = 10,
                    monitor: str = 'val_loss') -> list:
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
        logs_dir: Directory for training logs
        patience: Early stopping patience
        monitor: Metric to monitor
        
    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_save_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV logger
        CSVLogger(
            os.path.join(logs_dir, 'training_log.csv'),
            append=False
        )
    ]
    
    return callbacks

def train_model(model: keras.Model,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               epochs: int = 50,
               batch_size: int = 32,
               use_class_weights: bool = True,
               use_augmentation: bool = True,
               model_save_path: str = None,
               logs_dir: str = None) -> keras.callbacks.History:
    """
    Train a model
    
    Args:
        model: Keras model to train
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        use_class_weights: Whether to use class weights
        use_augmentation: Whether to use data augmentation
        model_save_path: Path to save best model
        logs_dir: Directory for logs
        
    Returns:
        Training history
    """
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
    
    # Create callbacks
    if model_save_path is None:
        model_save_path = 'models/saved_models/best_model.h5'
    if logs_dir is None:
        logs_dir = 'logs/training_logs'
    
    callbacks = create_callbacks(model_save_path, logs_dir)
    
    # Data augmentation
    if use_augmentation:
        from src.data.augmentation import create_train_generator
        train_gen = create_train_generator(X_train, y_train, batch_size, augment=True)
        steps_per_epoch = len(X_train) // batch_size
        
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    
    return history
