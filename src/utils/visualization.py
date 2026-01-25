"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Optional

def plot_images(images: np.ndarray, 
               labels: np.ndarray = None,
               class_names: List[str] = None,
               num_images: int = 9,
               figsize: tuple = (12, 12)) -> None:
    """
    Plot a grid of images
    
    Args:
        images: Array of images
        labels: Array of labels (optional)
        class_names: List of class names
        num_images: Number of images to plot
        figsize: Figure size
    """
    num_images = min(num_images, len(images))
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i in range(num_images):
        ax = axes[i]
        img = images[i]
        
        # Handle grayscale images
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.axis('off')
        
        if labels is not None:
            label = labels[i]
            title = class_names[label] if class_names else f'Class {label}'
            ax.set_title(title, fontsize=10)
    
    # Hide extra subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str] = None,
                         figsize: tuple = (8, 6)) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  class_names: List[str] = None,
                  figsize: tuple = (8, 6)) -> None:
    """
    Plot ROC curve (for binary or multi-class)
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        figsize: Figure size
    """
    n_classes = y_pred_proba.shape[1]
    
    plt.figure(figsize=figsize)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class: one-vs-rest
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = class_names[i] if class_names else f'Class {i}'
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_training_history(history, figsize: tuple = (15, 5)) -> None:
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Keras training history object
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels: np.ndarray,
                           class_names: List[str] = None,
                           figsize: tuple = (8, 6)) -> None:
    """
    Plot class distribution
    
    Args:
        labels: Array of labels
        class_names: List of class names
        figsize: Figure size
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(unique)), counts, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    if class_names:
        plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    else:
        plt.xticks(range(len(unique)), [f'Class {i}' for i in unique])
    
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()
