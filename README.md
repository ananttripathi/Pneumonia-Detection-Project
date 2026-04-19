---
title: Pneumonia Detection
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Pneumonia Detection from Chest X-Ray Images

CNN-based pneumonia detection from chest X-rays using EfficientNetB0 transfer learning. Includes DICOM handling, Streamlit deployment, and full MLOps pipeline via Hugging Face Hub.

## 🔗 Resources

| Resource | Link |
|----------|------|
| 🤗 Dataset | [ananttripathiak/pneumonia-detection-dataset](https://huggingface.co/datasets/ananttripathiak/pneumonia-detection-dataset) |
| 🤖 Model | [ananttripathiak/pneumonia-detection-model](https://huggingface.co/ananttripathiak/pneumonia-detection-model) |
| 🚀 Live App | [HF Space](https://huggingface.co/spaces/ananttripathiak/pneumonia-detection-space) |

## ⚡ Quick Start

```bash
git clone https://github.com/ananttripathi/Pneumonia-Detection-Project.git
cd Pneumonia-Detection-Project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks in order (`notebooks/01_data_overview.ipynb` → … → `05_transfer_learning.ipynb`) or start the Streamlit app: `streamlit run src/deployment/app.py`. See [QUICKSTART.md](QUICKSTART.md) and [Installation](#installation) for data setup and usage.

## Business Context

Pneumonia is one of the leading causes of morbidity and mortality worldwide, particularly affecting children under five years and elderly populations. According to the World Health Organization (WHO), pneumonia accounts for a significant percentage of deaths caused by infectious diseases. Early detection and timely treatment are critical to improving patient outcomes, yet current diagnostic methods present challenges.

### Current Diagnostic Challenges

The most common method for diagnosing pneumonia is through clinical evaluation combined with chest X-ray imaging. However, several challenges exist:

- **Limited Radiologist Availability**: Accurate interpretation of X-rays requires skilled radiologists, whose availability is limited in many regions, especially in rural or resource-constrained healthcare settings
- **Human Factors**: Even when radiologists are available, factors such as fatigue, high patient load, and human error can affect the accuracy and consistency of diagnosis
- **Healthcare Impact**: These challenges may lead to:
  - Delayed treatment
  - Misdiagnosis
  - Unnecessary use of antibiotics
  - Worsening patient outcomes
  - Strain on healthcare systems

### AI-Driven Solution

With the advancement of machine learning and deep learning, automated image analysis has emerged as a promising solution to support medical imaging tasks. Leveraging large datasets of chest X-ray images, AI-driven approaches can be trained to recognize pneumonia-related abnormalities in the lungs with high accuracy and consistency. Such systems can serve as decision-support tools for healthcare professionals, reducing diagnostic workload, improving accuracy, and providing timely interventions, particularly in areas with limited medical expertise.

## Objective

The main objective of this project is to develop an intelligent, automated system capable of detecting pneumonia from chest X-ray images using machine learning and deep learning techniques.

### System Goals

The system should aim to:

1. **Accurately classify** chest X-ray images into pneumonia-positive and pneumonia-negative cases
2. **Assist healthcare professionals** by providing a reliable second opinion that reduces diagnostic errors and variability
3. **Improve efficiency** by delivering faster diagnoses, enabling timely treatment, and reducing the burden on radiologists
4. **Enhance accessibility** by offering a scalable solution that can be deployed in hospitals, clinics, or rural healthcare centers with limited resources
5. **Support global health efforts** by contributing to early detection, lowering pneumonia-related mortality rates, and optimizing antibiotic usage

Ultimately, the solution aims to bridge the gap between limited medical expertise and growing healthcare demands, making pneumonia diagnosis more accurate, efficient, and accessible worldwide.

## Data Description

### Dataset Overview

The dataset contains chest X-ray images with the following characteristics:

- **Classes**: 
  - Pneumonia-positive
  - Pneumonia-negative (Normal)
  - Not Normal No Lung Opacity (abnormality present but not pneumonia)

### Special Classification Note

In the dataset, some features are labeled **"Not Normal No Lung Opacity"**. This extra third class indicates that while pneumonia was determined not to be present, there was nonetheless some type of abnormality on the image, and oftentimes this finding may mimic the appearance of true pneumonia.

### Image Format

**DICOM Original Images**: Medical images are stored in a special format called DICOM files (`*.dcm`). They contain a combination of:
- Header metadata
- Underlying raw image arrays for pixel data

## Evaluation Rubrics

### Interim Report (Total: 40 Points)

| Section | Description | Points |
|---------|-------------|--------|
| **Data Overview** | - Import the data<br>- Check the shape of the data | 6 |
| **Exploratory Data Analysis** | - Plot random images from each class and print their corresponding labels<br>- Check for class imbalance<br>- Key meaningful observations from EDA | 8 |
| **Data Preprocessing** | - Convert the RGB images to Grayscale<br>- Plot the images before and after the pre-processing steps<br>- Split the data into train, validation and test<br>- Apply the normalization | 10 |
| **Model Building** | - Define a CNN model from scratch<br>- Train the Model<br>- Check and comment on the performance of the model | 10 |
| **Business Report Quality** | - Adhere to the business report checklist | 6 |

### Final Report (Total: 60 Points)

| Section | Description | Points |
|---------|-------------|--------|
| **Data Overview** | - Import the data<br>- Check the shape of the data | 3 |
| **Exploratory Data Analysis** | - Plot random images from each class and print their corresponding labels<br>- Check for class imbalance<br>- Key meaningful observations from EDA | 3 |
| **Data Preprocessing** | - Convert the RGB images to Grayscale<br>- Plot the images before and after the pre-processing steps<br>- Split the data into train, validation and test<br>- Apply the normalization | 4 |
| **Model Building** | - Define a CNN model from scratch<br>- Train the Model<br>- Check and comment on the performance of the model | 5 |
| **Transfer Learning** | - Apply transfer learning using pre-trained CNN models (at least 2)<br>- Check and comment on the performance of the models<br>- Create new architectures using the above pre-trained CNNs and adding additional layers<br>- Check and comment on the performance of the models<br>- Compare and comment on the performance of all models built<br>- Choose the best model with a proper rationale<br>- Serialize the best model, re-load it, and make Inferences on a few images | 30 |
| **Model Deployment** | - Build a Streamlit or Gradio app where users can upload an image and see predicted class + probability<br>- Package app + model inside a Docker container for portability<br>- Deploy to a Hugging Face platform and make an inference | 5 |
| **Actionable Insights and Recommendations** | - Key takeaways for the business | 4 |
| **Business Report Quality** | - Adhere to the business report checklist | 6 |

---

## Project Structure
```
Pneumonia-Detection-Project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_cnn_from_scratch.ipynb
│   └── 05_transfer_learning.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── cnn_scratch.py
│   │   ├── transfer_learning.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── utils/
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   └── config.py
│   └── deployment/
│       └── app.py
├── models/
│   └── saved_models/
│       ├── best_model.h5
│       └── best_model.pkl
├── logs/
│   └── training_logs/
└── docs/
    ├── business_report.md
    └── technical_documentation.md
```

## Installation
```bash
# Clone the repository
git clone https://github.com/ananttripathi/Pneumonia-Detection-Project.git
cd Pneumonia-Detection-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
```python
from src.data.preprocessing import preprocess_images
from src.data.data_loader import load_dicom_images

# Load DICOM images
images, labels = load_dicom_images('data/raw/')

# Preprocess images
processed_images = preprocess_images(images, grayscale=True, normalize=True)
```

### 2. Model Training (CNN from Scratch)
```python
from src.models.cnn_scratch import build_cnn_model
from src.models.train import train_model

# Build CNN model
model = build_cnn_model(input_shape=(224, 224, 1), num_classes=3)

# Train model
history = train_model(
    model, 
    X_train, y_train, 
    X_val, y_val,
    epochs=50,
    batch_size=32
)
```

### 3. Transfer Learning
```python
from src.models.transfer_learning import build_transfer_model
from tensorflow.keras.applications import VGG16, ResNet50

# Build transfer learning model with VGG16
model_vgg = build_transfer_model(
    base_model=VGG16,
    input_shape=(224, 224, 3),
    num_classes=3,
    freeze_base=True
)

# Build transfer learning model with ResNet50
model_resnet = build_transfer_model(
    base_model=ResNet50,
    input_shape=(224, 224, 3),
    num_classes=3,
    freeze_base=True
)
```

### 4. Model Evaluation
```python
from src.models.evaluate import evaluate_model
from src.utils.metrics import plot_confusion_matrix, plot_roc_curve

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(y_test, predictions)

# Plot ROC curve
plot_roc_curve(y_test, predictions)
```

### 5. Making Predictions
```python
from src.models.predict import predict_pneumonia
from PIL import Image

# Load and predict
image = Image.open('test_image.jpg')
prediction, probability = predict_pneumonia(model, image)

print(f"Prediction: {prediction}")
print(f"Confidence: {probability:.2%}")
```

## Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** - Deep learning framework
- **PyTorch** (optional) - Alternative deep learning framework
- **OpenCV** - Image processing
- **Pydicom** - DICOM file handling
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Metrics and utilities
- **Streamlit / Gradio** - Web application deployment
- **Docker** - Containerization
- **Hugging Face Spaces** - Model hosting and deployment

## Deep Learning Models

### Custom CNN Architecture

Built from scratch with:
- Convolutional layers
- Max pooling layers
- Batch normalization
- Dropout for regularization
- Dense layers for classification

### Transfer Learning Models

Pre-trained models implemented (minimum 2):
- **VGG16** - Visual Geometry Group 16-layer network
- **ResNet50** - Residual Network with 50 layers
- **InceptionV3** - Inception architecture
- **EfficientNet** - Efficient architecture scaling
- **DenseNet** - Densely connected networks

## Key Features

- 🏥 **Medical Image Processing**: Specialized handling of DICOM files
- 🔍 **Advanced Preprocessing**: Grayscale conversion, normalization, augmentation
- 🧠 **Multiple CNN Architectures**: Custom and pre-trained models
- 📊 **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC
- 🎯 **Class Imbalance Handling**: Weighted loss functions and data augmentation
- 🚀 **Deployment Ready**: Streamlit/Gradio app with Docker support
- 📱 **User-Friendly Interface**: Upload image and get instant predictions
- 🐳 **Containerized**: Docker support for easy deployment

## Data Preprocessing Pipeline

### 1. DICOM Loading
- Read DICOM files with metadata
- Extract pixel arrays

### 2. Image Conversion
- Convert RGB to Grayscale
- Resize to standard dimensions (e.g., 224x224)

### 3. Normalization
- Pixel value scaling (0-1 or -1 to 1)
- Standardization (mean=0, std=1)

### 4. Data Augmentation
- Rotation
- Horizontal flip
- Zoom
- Brightness adjustment
- Contrast adjustment

### 5. Data Splitting
- Training set (70-80%)
- Validation set (10-15%)
- Test set (10-15%)

## Model Performance Metrics

The models are evaluated using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of correct positive predictions
- **Recall (Sensitivity)**: Ability to identify all positive cases
- **Specificity**: Ability to identify negative cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

## Deployment

### Streamlit App
```bash
# Run locally
streamlit run src/deployment/app.py
```

### Gradio App
```bash
# Run locally
python src/deployment/gradio_app.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t pneumonia-detection .

# Run container
docker run -p 8501:8501 pneumonia-detection
```

### Hugging Face Spaces

Deploy the Streamlit app to [Hugging Face Spaces](https://huggingface.co/spaces). Use the Docker/Streamlit template and upload `src/deployment/app.py`, `requirements.txt`, and your trained model. Example: `https://huggingface.co/spaces/<your-username>/pneumonia-detection`

## Sample Web Interface Features

- 📤 **Image Upload**: Drag-and-drop or browse to upload chest X-ray
- 🔮 **Real-time Prediction**: Instant classification results
- 📊 **Confidence Score**: Probability distribution across classes
- 🖼️ **Image Preview**: View uploaded image
- 📈 **Visualization**: Heatmap/Grad-CAM for interpretability
- 📝 **Recommendation**: Suggested next steps based on prediction

## Model Interpretability

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualize which regions of the X-ray influenced the model's decision
- Highlight areas of interest for pneumonia detection

### Feature Maps
- Display intermediate layer activations
- Understand what features the model learns

## Clinical Impact

### Benefits for Healthcare Providers

- ⚡ **Faster Diagnosis**: Reduce time from imaging to diagnosis
- 🎯 **Higher Accuracy**: Reduce false negatives and false positives
- 👨‍⚕️ **Radiologist Support**: Second opinion and decision support
- 📉 **Reduced Workload**: Automate preliminary screening
- 🌍 **Increased Access**: Deploy in resource-limited settings

### Patient Benefits

- ⏱️ **Timely Treatment**: Earlier intervention improves outcomes
- 💊 **Appropriate Antibiotic Use**: Reduce unnecessary prescriptions
- 🏥 **Better Outcomes**: Lower mortality and morbidity rates
- 💰 **Cost Savings**: Reduce complications and hospitalizations

## Results

*(Add your model performance results here)*

### Example Results Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Custom CNN | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| VGG16 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| ResNet50 | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| **Best Model** | **0.XX** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

## Future Enhancements

- 🔬 **Multi-class Classification**: Distinguish between bacterial and viral pneumonia
- 🌐 **Multi-modal Learning**: Incorporate clinical data with imaging
- 📱 **Mobile Application**: Deploy on smartphones for point-of-care use
- 🔄 **Continuous Learning**: Update model with new data
- 🏥 **Integration**: Connect with hospital PACS systems
- 🌍 **Multilingual Support**: Interface in multiple languages
- 📊 **Analytics Dashboard**: Track model performance over time

## Ethical Considerations

- ⚖️ **Not a Replacement**: Tool assists but doesn't replace radiologists
- 🔒 **Privacy**: HIPAA-compliant data handling
- ✅ **Validation**: Clinical validation required before deployment
- 🌈 **Bias**: Monitor for demographic and technical biases
- 📋 **Regulatory**: Comply with medical device regulations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for run instructions and how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).

**Suggested GitHub topics:** `machine-learning` `computer-vision` `healthcare` `deep-learning` `tensorflow` `medical-imaging` `pneumonia-detection` `transfer-learning`

## Author

**Co-author:** [ananttripathiak](mailto:ananttripathiak@gmail.com)

## Contact

For questions or support, please open a [GitHub Issue](https://github.com/ananttripathi/Pneumonia-Detection-Project/issues).

## Acknowledgments

- Dataset provided by [source]
- Medical experts for validation and guidance
- Open-source deep learning community
- Healthcare institutions for collaboration

## Disclaimer

⚠️ **This tool is for research and educational purposes only. It should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.**
