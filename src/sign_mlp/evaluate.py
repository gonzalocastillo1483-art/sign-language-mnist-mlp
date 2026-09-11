import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from sign_mlp.data import as_images, label_to_letter


def report_metrics(y_true, y_pred, index_to_label):
    """Return a text classification report."""
    labels = sorted(index_to_label)
    target_names = [label_to_letter(index_to_label[index]) for index in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )


def plot_history(history):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, index_to_label):
    """Plot a confusion matrix with class letters."""
    labels = sorted(index_to_label)
    names = [label_to_letter(index_to_label[index]) for index in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion")
    fig.tight_layout()
    return fig


def plot_prediction_examples(X, y_true, y_pred, index_to_label, correct=True, max_items=9):
    """Plot correct or incorrect predictions."""
    mask = y_true == y_pred if correct else y_true != y_pred
    indices = np.where(mask)[0][:max_items]
    images = as_images(X[indices])

    cols = 3
    rows = int(np.ceil(max(len(indices), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, image, idx in zip(axes, images, indices):
        real = label_to_letter(index_to_label[int(y_true[idx])])
        pred = label_to_letter(index_to_label[int(y_pred[idx])])
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Real: {real} | Pred: {pred}")
        ax.axis("off")

    title = "Ejemplos correctos" if correct else "Ejemplos mal clasificados"
    fig.suptitle(title)
    fig.tight_layout()
    return fig
