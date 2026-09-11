from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_SIZE = 28
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE


def label_to_letter(label: int) -> str:
    """Convert Sign Language MNIST numeric labels to letters."""
    label = int(label)
    if 0 <= label <= 25:
        return chr(ord("A") + label)
    return str(label)


def load_sign_mnist(raw_dir: str | Path):
    """Load Sign Language MNIST CSV files from data/raw."""
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "sign_mnist_train.csv"
    test_path = raw_dir / "sign_mnist_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing sign_mnist_train.csv or sign_mnist_test.csv. "
            "Run scripts/prepare_data.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    return X_train, y_train, X_test, y_test


def make_label_mapping(*label_arrays):
    """Create contiguous class indices from original dataset labels."""
    labels = sorted({int(label) for labels in label_arrays for label in labels})
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


def remap_labels(y, label_to_index):
    """Map original labels to contiguous indices for model training."""
    return np.array([label_to_index[int(label)] for label in y], dtype=np.int64)


def as_images(X):
    """Reshape flattened vectors into 28x28 images for visualization."""
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
