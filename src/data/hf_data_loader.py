"""
Load dataset from Hugging Face Hub — streams DICOM images in batches
to avoid loading 26k files into memory at once.
"""
import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import tensorflow as tf
from huggingface_hub import snapshot_download
from tensorflow import keras
from typing import Tuple, Optional

DATASET_REPO = "ananttripathiak/pneumonia-detection-dataset"
IMG_SIZE = (224, 224)
CLASS_TO_IDX = {"Normal": 0, "Lung Opacity": 1, "No Lung Opacity / Not Normal": 2}


def download_dataset(local_dir: str = "/tmp/pneumonia-data", token: str = None) -> str:
    print(f"Downloading dataset from {DATASET_REPO}...")
    path = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=local_dir,
        token=token,
    )
    print(f"Dataset downloaded to {path}")
    return path


def load_dicom_as_rgb(dcm_path: str) -> Optional[np.ndarray]:
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
        if img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error loading {dcm_path}: {e}")
        return None


def build_dataframe(dataset_dir: str) -> pd.DataFrame:
    # Support both HF Hub layout (data/) and local Downloads layout (root)
    if os.path.exists(os.path.join(dataset_dir, "data", "stage_2_train_labels.csv")):
        labels_path = os.path.join(dataset_dir, "data", "stage_2_train_labels.csv")
        class_path = os.path.join(dataset_dir, "data", "stage_2_detailed_class_info.csv")
        train_dir = os.path.join(dataset_dir, "data", "train_images")
    else:
        labels_path = os.path.join(dataset_dir, "stage_2_train_labels.csv")
        class_path = os.path.join(dataset_dir, "stage_2_detailed_class_info.csv")
        train_dir = os.path.join(dataset_dir, "stage_2_train_images")

    labels_df = pd.read_csv(labels_path)
    class_df = pd.read_csv(class_path)
    merged = labels_df.merge(class_df[["patientId", "class"]], on="patientId", how="left")
    merged["label"] = merged["class"].map(CLASS_TO_IDX).fillna(0).astype(int)
    merged["filepath"] = merged["patientId"].apply(
        lambda pid: os.path.join(train_dir, f"{pid}.dcm")
    )
    merged = merged[merged["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Found {len(merged)} images with labels")
    print(merged["class"].value_counts())
    return merged


class DICOMSequence(keras.utils.Sequence):
    """Keras Sequence that loads DICOM files batch-by-batch."""

    def __init__(self, df: pd.DataFrame, batch_size: int = 32, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.df))

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]

        images, labels = [], []
        for _, row in batch.iterrows():
            img = load_dicom_as_rgb(row["filepath"])
            if img is not None:
                if self.augment:
                    img = self._augment(img)
                images.append(img)
                labels.append(row["label"])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        angle = np.random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img
