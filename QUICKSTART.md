# Quick Start Guide

## Setup

1. **Clone the repository** (if not already done):
```bash
cd /Users/ananttripathi/Desktop
git clone https://github.com/ananttripathi/Pneumonia-Detection-Project.git
cd Pneumonia-Detection-Project
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify data location**:
   - Ensure the `pneumonia_files` folder is on your Desktop
   - It should contain:
     - `stage_2_train_images.zip`
     - `stage_2_test_images.zip`
     - `stage_2_train_labels.csv`
     - `stage_2_detailed_class_info.csv`

## Running the Project

### Option 1: Using Jupyter Notebooks (Recommended for Development)

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Run notebooks in order**:
   - `notebooks/01_data_overview.ipynb` - Explore the dataset
   - `notebooks/02_eda.ipynb` - Exploratory data analysis
   - `notebooks/03_preprocessing.ipynb` - Preprocess images
   - `notebooks/04_cnn_from_scratch.ipynb` - Train custom CNN
   - `notebooks/05_transfer_learning.ipynb` - Train transfer learning models

### Option 2: Using Python Scripts

You can also use the modules directly in Python:

```python
from src.data.data_loader import load_dicom_images
from src.data.preprocessing import preprocess_images, split_data
from src.models.cnn_scratch import build_cnn_model
from src.models.train import train_model

# Load and preprocess data
pneumonia_files_path = '/Users/ananttripathi/Desktop/pneumonia_files'
images, labels = load_dicom_images(
    zip_path=f'{pneumonia_files_path}/stage_2_train_images.zip',
    labels_csv=f'{pneumonia_files_path}/stage_2_train_labels.csv',
    class_info_csv=f'{pneumonia_files_path}/stage_2_detailed_class_info.csv'
)

processed = preprocess_images(images, grayscale=True, normalize=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed, labels)

# Build and train model
model = build_cnn_model(input_shape=(224, 224, 1), num_classes=3)
history = train_model(model, X_train, y_train, X_val, y_val)
```

## Deployment

### Local Streamlit App

1. **Train a model first** (using notebooks or scripts)

2. **Run Streamlit app**:
```bash
streamlit run src/deployment/app.py
```

3. **Open browser** to `http://localhost:8501`

### Docker Deployment

1. **Build Docker image**:
```bash
docker build -t pneumonia-detection .
```

2. **Run container**:
```bash
docker run -p 8501:8501 pneumonia-detection
```

3. **Access app** at `http://localhost:8501`

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload the following files:
   - `src/deployment/app.py`
   - `requirements.txt`
   - `models/saved_models/best_model.h5` (trained model)
3. Configure the Space to use Streamlit
4. Deploy!

## Project Structure

```
Pneumonia-Detection-Project/
├── notebooks/          # Jupyter notebooks for analysis and training
├── src/                # Source code modules
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model definitions and training
│   ├── utils/         # Utility functions
│   └── deployment/    # Streamlit app
├── models/            # Saved models
├── logs/              # Training logs
├── docs/              # Documentation
└── data/              # Data directories
```

## Important Notes

1. **Data Location**: The code expects data in `/Users/ananttripathi/Desktop/pneumonia_files/`
   - Update paths in notebooks if your data is elsewhere

2. **Memory Requirements**: 
   - Loading all images at once may require significant RAM
   - Consider using `max_images` parameter for testing

3. **Model Training**:
   - Training can take several hours depending on your hardware
   - Use GPU if available for faster training

4. **First Run**:
   - Pre-trained models will be downloaded automatically
   - This may take some time on first run

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

### DICOM Loading Issues
- Ensure `pydicom` is installed: `pip install pydicom`
- Check that zip files are not corrupted

### Memory Errors
- Reduce batch size in training
- Use `max_images` parameter when loading data
- Process data in smaller batches

### Model Not Found
- Train a model first using the notebooks
- Check that model is saved in `models/saved_models/best_model.h5`

## Next Steps

1. Run the notebooks to train models
2. Evaluate model performance
3. Deploy using Streamlit or Docker
4. Refer to `docs/business_report.md` and `docs/technical_documentation.md` for detailed information

## Support

For issues or questions, refer to:
- Technical Documentation: `docs/technical_documentation.md`
- Business Report: `docs/business_report.md`
- README: `README.md`
