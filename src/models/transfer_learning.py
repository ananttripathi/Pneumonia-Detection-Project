"""
Transfer learning models using pre-trained CNNs
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB0, DenseNet121
from typing import Tuple, Callable

def build_transfer_model(base_model_class: Callable,
                        input_shape: Tuple[int, int, int] = (224, 224, 3),
                        num_classes: int = 3,
                        freeze_base: bool = True,
                        dropout_rate: float = 0.5,
                        dense_units: int = 512) -> Model:
    """
    Build a transfer learning model using a pre-trained base
    
    Args:
        base_model_class: Pre-trained model class (VGG16, ResNet50, etc.)
        input_shape: Shape of input images
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model weights
        dropout_rate: Dropout rate
        dense_units: Number of units in dense layer
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained base model
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    else:
        # Fine-tune last few layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for specific models
    if base_model_class == VGG16:
        x = base_model(inputs, training=False)
    elif base_model_class in [ResNet50, InceptionV3, DenseNet121]:
        x = base_model(inputs, training=False)
    elif base_model_class == EfficientNetB0:
        x = base_model(inputs, training=False)
    else:
        x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Additional dense layers
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_vgg16_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                     num_classes: int = 3,
                     freeze_base: bool = True) -> Model:
    """Build VGG16 transfer learning model"""
    return build_transfer_model(
        VGG16, input_shape, num_classes, freeze_base
    )

def build_resnet50_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                        num_classes: int = 3,
                        freeze_base: bool = True) -> Model:
    """Build ResNet50 transfer learning model"""
    return build_transfer_model(
        ResNet50, input_shape, num_classes, freeze_base
    )

def build_inceptionv3_model(input_shape: Tuple[int, int, int] = (299, 299, 3),
                            num_classes: int = 3,
                            freeze_base: bool = True) -> Model:
    """Build InceptionV3 transfer learning model"""
    return build_transfer_model(
        InceptionV3, input_shape, num_classes, freeze_base
    )

def build_efficientnet_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 3,
                            freeze_base: bool = True) -> Model:
    """Build EfficientNetB0 transfer learning model"""
    return build_transfer_model(
        EfficientNetB0, input_shape, num_classes, freeze_base
    )

def build_densenet_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 3,
                         freeze_base: bool = True) -> Model:
    """Build DenseNet121 transfer learning model"""
    return build_transfer_model(
        DenseNet121, input_shape, num_classes, freeze_base
    )
