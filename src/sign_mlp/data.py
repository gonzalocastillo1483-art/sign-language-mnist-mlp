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


def read_sign_mnist_frames(raw_dir: str | Path):
    """Read original Sign Language MNIST CSV files as pandas DataFrames."""
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
    return train_df, test_df


def load_sign_mnist(raw_dir: str | Path):
    """Load Sign Language MNIST CSV files with normalized pixel values."""
    train_df, test_df = read_sign_mnist_frames(raw_dir)

    X_train = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    return X_train, y_train, X_test, y_test


def make_label_mapping(*label_arrays):
    """Create contiguous class indices from original dataset labels."""
    labels = sorted({int(label) for array in label_arrays for label in array})
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


def remap_labels(y, label_to_index):
    """Map original labels to contiguous indices for model training."""
    return np.array([label_to_index[int(label)] for label in y], dtype=np.int64)


def as_images(X):
    """Reshape flattened vectors into 28x28 images for visualization."""
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)


def stratified_train_validation_split(X, y, validation_size=0.2, random_state=42):
    """Split arrays while keeping a similar class distribution in both sets."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    validation_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0].copy()
        rng.shuffle(label_indices)
        validation_count = max(1, int(round(len(label_indices) * validation_size)))
        validation_indices.extend(label_indices[:validation_count])
        train_indices.extend(label_indices[validation_count:])

    train_indices = np.array(train_indices, dtype=np.int64)
    validation_indices = np.array(validation_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return X[train_indices], X[validation_indices], y[train_indices], y[validation_indices]


def prepare_sign_mnist_data(raw_dir: str | Path, validation_size=0.2, random_state=42):
    """Load, normalize, remap labels and create a stratified validation split."""
    X_train_full, y_train_full, X_test, y_test = load_sign_mnist(raw_dir)

    label_to_index, index_to_label = make_label_mapping(y_train_full, y_test)
    y_train_full_idx = remap_labels(y_train_full, label_to_index)
    y_test_idx = remap_labels(y_test, label_to_index)

    X_train, X_val, y_train, y_val = stratified_train_validation_split(
        X_train_full,
        y_train_full_idx,
        validation_size=validation_size,
        random_state=random_state,
    )

    metadata = {
        "image_size": IMAGE_SIZE,
        "input_dim": INPUT_DIM,
        "num_classes": len(index_to_label),
        "class_labels": [index_to_label[index] for index in sorted(index_to_label)],
        "class_names": [
            label_to_letter(index_to_label[index]) for index in sorted(index_to_label)
        ],
        "validation_size": validation_size,
        "random_state": random_state,
    }

    return {
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test_idx,
        "y_train_full_idx": y_train_full_idx,
        "y_test_original": y_test,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "metadata": metadata,
    }
