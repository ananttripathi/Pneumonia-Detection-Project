"""
End-to-end training pipeline — downloads data from HF Hub, trains
EfficientNetB0, and pushes the best model back to HF Hub.

Usage:
    python run_training.py --token YOUR_HF_TOKEN
    python run_training.py --token YOUR_HF_TOKEN --epochs 30 --batch-size 32
"""
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from huggingface_hub import HfApi, create_repo
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from src.data.hf_data_loader import download_dataset, build_dataframe, DICOMSequence

MODEL_REPO = "ananttripathiak/pneumonia-detection-model"
MODEL_SAVE_PATH = "models/saved_models/best_model.h5"
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("logs/training_logs", exist_ok=True)


def build_model(num_classes: int = 3) -> Model:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="/tmp/pneumonia-data")
    args = parser.parse_args()

    # 1 — Download dataset
    dataset_dir = download_dataset(local_dir=args.data_dir, token=args.token)

    # 2 — Build dataframe of filepaths + labels
    df = build_dataframe(dataset_dir)

    # 3 — Train / val split (stratified)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 4 — Class weights for imbalanced data
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(train_df["label"]), y=train_df["label"]
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Class weights: {class_weights}")

    # 5 — Data generators
    train_gen = DICOMSequence(train_df, batch_size=args.batch_size, augment=True)
    val_gen = DICOMSequence(val_df, batch_size=args.batch_size, augment=False)

    # 6 — Build model
    model = build_model()

    # 7 — Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    # 8 — Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 9 — Upload model to HF Hub
    print(f"\nUploading model to {MODEL_REPO}...")
    api = HfApi(token=args.token)
    create_repo(repo_id=MODEL_REPO, repo_type="model", token=args.token, exist_ok=True)
    api.upload_file(
        path_or_fileobj=MODEL_SAVE_PATH,
        path_in_repo="best_model.h5",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")

    # 10 — Print final metrics
    val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
