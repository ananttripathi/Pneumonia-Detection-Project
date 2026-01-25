"""
Model evaluation module
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
from typing import Dict, Tuple
import tensorflow as tf

def evaluate_model(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (one-vs-rest for multi-class)
    try:
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

def get_confusion_matrix(model: tf.keras.Model,
                        X_test: np.ndarray,
                        y_test: np.ndarray) -> np.ndarray:
    """
    Get confusion matrix
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        
    Returns:
        Confusion matrix
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    return cm

def get_classification_report(model: tf.keras.Model,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             target_names: list = None) -> str:
    """
    Get detailed classification report
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        target_names: Names of classes
        
    Returns:
        Classification report as string
    """
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    return report
