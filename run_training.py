"""
End-to-end training pipeline — downloads data from HF Hub (or uses local files),
trains EfficientNetB3, and pushes the best model back to HF Hub.

Usage (local data — dataset already on disk):
    python run_training.py --token YOUR_HF_TOKEN --local-data /Users/ananttripathi/Downloads

Usage (download from HF Hub):
    python run_training.py --token YOUR_HF_TOKEN
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

sys.path.insert(0, os.path.dirname(__file__))
from src.data.hf_data_loader import download_dataset, build_dataframe, DICOMSequence
from src.model import build_model, IMG_SIZE

MODEL_REPO = "ananttripathiak/pneumonia-detection-model"
WEIGHTS_SAVE_PATH = "models/saved_models/best_model.weights.h5"
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("logs/training_logs", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default="/tmp/pneumonia-data", help="HF Hub download dir")
    parser.add_argument("--local-data", default=None,
                        help="Use local data dir instead of HF Hub download (e.g. /Users/you/Downloads)")
    args = parser.parse_args()

    # 1 — Get dataset (local or HF Hub)
    if args.local_data:
        print(f"Using local data from {args.local_data}")
        dataset_dir = args.local_data
    else:
        dataset_dir = download_dataset(local_dir=args.data_dir, token=args.token)

    # 2 — Build dataframe of filepaths + labels
    df = build_dataframe(dataset_dir, img_size=IMG_SIZE)

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
    train_gen = DICOMSequence(train_df, batch_size=args.batch_size, augment=True, img_size=IMG_SIZE)
    val_gen = DICOMSequence(val_df, batch_size=args.batch_size, augment=False, img_size=IMG_SIZE)

    # 6 — Build model
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    # 7 — Cosine decay LR schedule
    total_steps = len(train_gen) * args.epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    model.optimizer.learning_rate = lr_schedule

    # 8 — Callbacks (save weights only — avoids Keras version serialization issues)
    callbacks = [
        ModelCheckpoint(WEIGHTS_SAVE_PATH, monitor="val_accuracy", save_best_only=True,
                        save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1),
    ]

    # 9 — Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 10 — Upload weights to HF Hub
    print(f"\nUploading model weights to {MODEL_REPO}...")
    api = HfApi(token=args.token)
    create_repo(repo_id=MODEL_REPO, repo_type="model", token=args.token, exist_ok=True)
    api.upload_file(
        path_or_fileobj=WEIGHTS_SAVE_PATH,
        path_in_repo="best_model.weights.h5",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")

    # 11 — Print final metrics
    val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
