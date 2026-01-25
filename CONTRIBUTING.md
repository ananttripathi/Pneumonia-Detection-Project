# Contributing to Pneumonia Detection Project

Thanks for your interest in this project. This repository implements CNN-based pneumonia detection from chest X-rays using custom architectures and transfer learning.

## Quick links

- **Setup & run:** [README – Installation](README.md#installation) · [QUICKSTART.md](QUICKSTART.md)
- **Technical details:** [docs/technical_documentation.md](docs/technical_documentation.md)
- **Business context:** [docs/business_report.md](docs/business_report.md)

## How to run

1. **Clone and install**
   ```bash
   git clone https://github.com/ananttripathi/Pneumonia-Detection-Project.git
   cd Pneumonia-Detection-Project
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data**
   - Place chest X-ray data in `data/raw/` (see [QUICKSTART.md](QUICKSTART.md) for layout and CSV labels).
   - Use `stage_2_train_images.zip`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` as referenced in the project.

3. **Notebooks (recommended)**
   - Run in order: `01_data_overview` → `03_preprocessing` → `04_cnn_from_scratch` → `05_transfer_learning`.

4. **Streamlit app**
   - After training a model: `streamlit run src/deployment/app.py`

5. **Docker**
   - `docker build -t pneumonia-detection .` then `docker run -p 8501:8501 pneumonia-detection`

## Contributing

- Open a [GitHub Issue](https://github.com/ananttripathi/Pneumonia-Detection-Project/issues) for bugs or ideas.
- Pull requests welcome. Ensure notebooks and scripts run with the documented setup.

## Disclaimer

This tool is for **research and education only**. It is not a medical device. Always rely on qualified healthcare professionals for diagnosis and treatment.
