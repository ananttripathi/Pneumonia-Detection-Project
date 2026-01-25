"""
Data loading module for DICOM images and CSV labels
"""
import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from typing import Tuple, List, Optional
import cv2
from tqdm import tqdm
import zipfile
import tempfile

def load_dicom_image(dicom_path: str) -> np.ndarray:
    """
    Load a single DICOM image and return pixel array
    
    Args:
        dicom_path: Path to DICOM file
        
    Returns:
        Pixel array as numpy array
    """
    try:
        dicom_file = pydicom.dcmread(dicom_path)
        pixel_array = dicom_file.pixel_array
        
        # Normalize to 0-255 range if needed
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        return pixel_array
    except Exception as e:
        print(f"Error loading DICOM file {dicom_path}: {str(e)}")
        return None

def load_labels_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load labels from CSV file
    
    Args:
        csv_path: Path to CSV file with labels
        
    Returns:
        DataFrame with patient IDs and labels
    """
    labels_df = pd.read_csv(csv_path)
    return labels_df

def load_images_from_zip(zip_path: str, patient_ids: List[str] = None, max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load DICOM images from zip file
    
    Args:
        zip_path: Path to zip file containing DICOM files
        patient_ids: Optional list of patient IDs to filter
        max_images: Maximum number of images to load (for testing)
        
    Returns:
        Tuple of (images, patient_ids)
    """
    images = []
    loaded_patient_ids = []
    
    print(f"Loading images from {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of DICOM files in zip
        dicom_files = [f for f in zip_ref.namelist() if f.endswith('.dcm')]
        
        if patient_ids:
            # Filter by patient IDs
            dicom_files = [f for f in dicom_files if any(pid in f for pid in patient_ids)]
        
        if max_images:
            dicom_files = dicom_files[:max_images]
        
        print(f"Found {len(dicom_files)} DICOM files in zip")
        
        # Extract to temporary directory and load
        with tempfile.TemporaryDirectory() as temp_dir:
            for dicom_file in tqdm(dicom_files):
                try:
                    # Extract file
                    zip_ref.extract(dicom_file, temp_dir)
                    dicom_path = os.path.join(temp_dir, dicom_file)
                    
                    # Load image
                    image = load_dicom_image(dicom_path)
                    
                    if image is not None:
                        images.append(image)
                        # Extract patient ID from filename
                        patient_id = os.path.basename(dicom_file).replace('.dcm', '')
                        loaded_patient_ids.append(patient_id)
                except Exception as e:
                    print(f"Error processing {dicom_file}: {str(e)}")
                    continue
    
    return images, loaded_patient_ids

def load_images_from_directory(directory: str, patient_ids: List[str] = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load DICOM images from directory
    
    Args:
        directory: Directory containing DICOM files
        patient_ids: Optional list of patient IDs to filter
        
    Returns:
        Tuple of (images, patient_ids)
    """
    images = []
    loaded_patient_ids = []
    
    # Get all DICOM files
    dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    
    if patient_ids:
        # Filter by patient IDs
        dicom_files = [f for f in dicom_files if any(pid in f for pid in patient_ids)]
    
    print(f"Loading {len(dicom_files)} DICOM files...")
    
    for dicom_file in tqdm(dicom_files):
        dicom_path = os.path.join(directory, dicom_file)
        image = load_dicom_image(dicom_path)
        
        if image is not None:
            images.append(image)
            # Extract patient ID from filename (assuming format: patientId.dcm)
            patient_id = dicom_file.replace('.dcm', '')
            loaded_patient_ids.append(patient_id)
    
    return images, loaded_patient_ids

def create_label_mapping(labels_df: pd.DataFrame, class_info_df: pd.DataFrame) -> dict:
    """
    Create mapping from patient ID to class label
    
    Args:
        labels_df: DataFrame with patient IDs and Target
        class_info_df: DataFrame with patient IDs and class names
        
    Returns:
        Dictionary mapping patient ID to class index
    """
    # Merge the two dataframes
    merged_df = labels_df.merge(class_info_df, on='patientId', how='left')
    
    # Create mapping
    label_mapping = {}
    class_to_idx = {
        'Normal': 0,
        'Lung Opacity': 1,
        'No Lung Opacity / Not Normal': 2
    }
    
    for _, row in merged_df.iterrows():
        patient_id = row['patientId']
        class_name = row['class']
        label_mapping[patient_id] = class_to_idx.get(class_name, 0)
    
    return label_mapping

def load_dicom_images(data_dir: str = None, 
                     zip_path: str = None,
                     labels_csv: str = None, 
                     class_info_csv: str = None,
                     max_images: Optional[int] = None,
                     patient_ids: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to load DICOM images and labels
    
    Args:
        data_dir: Directory containing DICOM files (optional if zip_path provided)
        zip_path: Path to zip file containing DICOM files (optional if data_dir provided)
        labels_csv: Path to labels CSV file
        class_info_csv: Path to class info CSV file
        max_images: Maximum number of images to load (for testing)
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    # Load images from directory or zip
    if zip_path:
        images, loaded_patient_ids = load_images_from_zip(zip_path, patient_ids=patient_ids, max_images=max_images)
    elif data_dir:
        images, loaded_patient_ids = load_images_from_directory(data_dir, patient_ids=patient_ids)
    else:
        raise ValueError("Either data_dir or zip_path must be provided")
    
    labels = None
    if labels_csv and class_info_csv:
        labels_df = load_labels_from_csv(labels_csv)
        class_info_df = load_labels_from_csv(class_info_csv)
        label_mapping = create_label_mapping(labels_df, class_info_df)
        
        # Map patient IDs to labels
        labels = [label_mapping.get(pid, 0) for pid in loaded_patient_ids]
        labels = np.array(labels)
    
    images = np.array(images)
    
    return images, labels
