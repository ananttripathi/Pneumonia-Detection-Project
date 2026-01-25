"""
Image preprocessing module
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale
    
    Args:
        image: Input image (can be RGB or already grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # RGB to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    
    return gray

def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target (height, width)
        
    Returns:
        Resized image
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

def normalize_image(image: np.ndarray, method: str = 'min_max') -> np.ndarray:
    """
    Normalize image pixel values
    
    Args:
        image: Input image
        method: Normalization method ('min_max' or 'standard')
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    if method == 'min_max':
        # Normalize to [0, 1]
        if image.max() > 1:
            image = image / 255.0
    elif method == 'standard':
        # Standardize (mean=0, std=1)
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
    elif method == 'tanh':
        # Normalize to [-1, 1]
        image = (image / 255.0) * 2.0 - 1.0
    
    return image

def preprocess_images(images: np.ndarray, 
                     grayscale: bool = True,
                     resize: bool = True,
                     target_size: Tuple[int, int] = (224, 224),
                     normalize: bool = True,
                     normalization_method: str = 'min_max') -> np.ndarray:
    """
    Preprocess a batch of images
    
    Args:
        images: Array of images
        grayscale: Whether to convert to grayscale
        resize: Whether to resize images
        target_size: Target size for resizing
        normalize: Whether to normalize
        normalization_method: Method for normalization
        
    Returns:
        Preprocessed images
    """
    processed_images = []
    
    for img in images:
        # Convert to grayscale if needed
        if grayscale:
            img = convert_to_grayscale(img)
        
        # Resize if needed
        if resize:
            img = resize_image(img, target_size)
        
        # Normalize if needed
        if normalize:
            img = normalize_image(img, normalization_method)
        
        # Add channel dimension if grayscale
        if grayscale and len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        processed_images.append(img)
    
    return np.array(processed_images)

def split_data(X: np.ndarray, 
               y: np.ndarray,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Features (images)
        y: Labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining after test split)
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
