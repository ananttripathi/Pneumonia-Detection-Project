# Technical Documentation: Pneumonia Detection Project

## Project Overview

This document provides technical documentation for the Pneumonia Detection from Chest X-Ray Images project.

## Architecture

### System Components

1. **Data Loading Module** (`src/data/data_loader.py`)
   - Handles DICOM file loading from zip archives
   - Creates label mappings from CSV files
   - Supports batch loading and filtering

2. **Preprocessing Module** (`src/data/preprocessing.py`)
   - Grayscale conversion
   - Image resizing
   - Normalization (min-max, standard, tanh)
   - Train/validation/test splitting

3. **Augmentation Module** (`src/data/augmentation.py`)
   - Rotation, shifting, zooming
   - Horizontal flipping
   - Brightness adjustment
   - Real-time augmentation during training

4. **Model Architectures**:
   - **Custom CNN** (`src/models/cnn_scratch.py`): 4 convolutional blocks with batch normalization and dropout
   - **Transfer Learning** (`src/models/transfer_learning.py`): VGG16, ResNet50, InceptionV3, EfficientNet, DenseNet

5. **Training Module** (`src/models/train.py`)
   - Class weight calculation for imbalanced data
   - Callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
   - CSV logging

6. **Evaluation Module** (`src/models/evaluate.py`)
   - Comprehensive metrics calculation
   - Confusion matrix
   - Classification report

7. **Deployment** (`src/deployment/app.py`)
   - Streamlit web interface
   - Image upload and prediction
   - Probability visualization

## Data Pipeline

### Input Format
- DICOM files (`.dcm`) in zip archives
- CSV files with patient IDs and labels

### Processing Steps
1. Extract DICOM files from zip
2. Load pixel arrays
3. Convert to grayscale
4. Resize to 224x224
5. Normalize to [0, 1]
6. Split into train/val/test (70/15/15)

## Model Details

### Custom CNN Architecture

```
Input (224, 224, 1)
  ↓
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
  ↓
Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
  ↓
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
  ↓
Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
  ↓
Flatten
  ↓
Dense(512) + BatchNorm + Dropout(0.5)
  ↓
Dense(256) + BatchNorm + Dropout(0.5)
  ↓
Dense(3, softmax)
```

### Transfer Learning Models

- **Base Models**: Pre-trained on ImageNet
- **Fine-tuning**: Option to freeze or fine-tune base layers
- **Top Layers**: GlobalAveragePooling2D + Dense layers

## Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Class Weights**: Computed automatically for imbalanced data
- **Data Augmentation**: Enabled during training

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: One-vs-rest for multi-class

## Deployment

### Local Deployment

```bash
streamlit run src/deployment/app.py
```

### Docker Deployment

```bash
docker build -t pneumonia-detection .
docker run -p 8501:8501 pneumonia-detection
```

### Hugging Face Spaces

1. Create a new Space
2. Upload model and app files
3. Configure `requirements.txt`
4. Deploy

## File Structure

```
pneumonia-detection/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_cnn_from_scratch.ipynb
│   └── 05_transfer_learning.ipynb
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── deployment/
├── models/
│   └── saved_models/
├── logs/
│   └── training_logs/
└── docs/
```

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- TensorFlow 2.10+
- Keras 2.10+
- OpenCV
- Pydicom
- Streamlit
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## Usage Examples

### Loading and Preprocessing Data

```python
from src.data.data_loader import load_dicom_images
from src.data.preprocessing import preprocess_images, split_data

images, labels = load_dicom_images(zip_path='path/to/images.zip')
processed = preprocess_images(images, grayscale=True, normalize=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed, labels)
```

### Training a Model

```python
from src.models.cnn_scratch import build_cnn_model
from src.models.train import train_model

model = build_cnn_model(input_shape=(224, 224, 1), num_classes=3)
history = train_model(model, X_train, y_train, X_val, y_val)
```

### Making Predictions

```python
from src.models.predict import predict_pneumonia

prediction, confidence, probabilities = predict_pneumonia(model, 'image.dcm')
```

## Performance Optimization

1. **Data Loading**: Use generators for large datasets
2. **Model Training**: Use mixed precision training
3. **Inference**: Optimize model for deployment (TensorFlow Lite)
4. **Caching**: Cache preprocessed data

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use data generators
2. **DICOM Loading Errors**: Ensure pydicom is properly installed
3. **Model Loading Errors**: Check model path and TensorFlow version compatibility

## Future Enhancements

1. Multi-class classification (bacterial vs viral pneumonia)
2. Multi-modal learning (combine imaging with clinical data)
3. Mobile deployment
4. Continuous learning pipeline
5. Integration with PACS systems
