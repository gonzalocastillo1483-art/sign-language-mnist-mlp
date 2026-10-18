"""Train and evaluate the EP2 CNN, then export reproducible evidence."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sign_mlp.cnn import build_cnn, set_global_seed, training_callbacks
from sign_mlp.data import prepare_sign_mnist_cnn_data
from sign_mlp.evaluate import (
    plot_confusion,
    plot_feature_maps,
    plot_gradcam_examples,
    plot_history,
    plot_prediction_examples,
)


SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def model_summary_text(model):
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def plot_model_comparison(comparison):
    """Compare test accuracy and macro F1 on the same percentage scale."""
    labels = comparison["model"].tolist()
    accuracy = (comparison["test_accuracy"] * 100).tolist()
    f1_macro = (comparison["test_f1_macro"] * 100).tolist()
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_accuracy = ax.bar(x - width / 2, accuracy, width, label="Accuracy test")
    bars_f1 = ax.bar(x + width / 2, f1_macro, width, label="F1 macro test")
    ax.set_ylabel("Porcentaje")
    ax.set_ylim(0, 105)
    ax.set_xticks(x, labels)
    ax.set_title("Comparacion experimental MLP y CNN")
    ax.legend()
    ax.bar_label(bars_accuracy, fmt="%.2f%%", padding=3)
    ax.bar_label(bars_f1, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    return fig


def main():
    set_global_seed(SEED)

    data = prepare_sign_mnist_cnn_data(
        ROOT / "data" / "raw",
        validation_size=0.2,
        random_state=SEED,
    )
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    index_to_label = data["index_to_label"]

    model = build_cnn(
        input_shape=data["metadata"]["input_shape"],
        num_classes=data["metadata"]["num_classes"],
        learning_rate=LEARNING_RATE,
    )

    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=training_callbacks(patience=4),
        verbose=2,
    )
    fit_seconds = time.perf_counter() - started

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    precision_macro = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)
    report_text = classification_report(
        y_test,
        predictions,
        target_names=data["metadata"]["class_names"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        predictions,
        target_names=data["metadata"]["class_names"],
        zero_division=0,
        output_dict=True,
    )

    reports_dir = ROOT / "reports"
    images_dir = ROOT / "images"
    models_dir = ROOT / "models"
    for directory in (reports_dir, images_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history.history).to_csv(
        reports_dir / "cnn_history.csv", index_label="epoch"
    )
    (reports_dir / "cnn_classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    (reports_dir / "cnn_model_summary.txt").write_text(
        model_summary_text(model),
        encoding="utf-8",
    )

    per_class = {
        name: values
        for name, values in report_dict.items()
        if name in data["metadata"]["class_names"]
    }
    ordered_classes = sorted(
        per_class,
        key=lambda name: per_class[name]["f1-score"],
    )
    metrics = {
        "model": "CNN_EP2",
        "seed": SEED,
        "input_shape": list(data["metadata"]["input_shape"]),
        "num_classes": data["metadata"]["num_classes"],
        "train_images": int(len(X_train)),
        "validation_images": int(len(X_val)),
        "test_images": int(len(X_test)),
        "epochs_requested": EPOCHS,
        "epochs_completed": int(len(history.history["loss"])),
        "batch_size": BATCH_SIZE,
        "learning_rate_initial": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "sparse_categorical_crossentropy",
        "parameters": int(model.count_params()),
        "fit_seconds": fit_seconds,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_check": float(accuracy_score(y_test, predictions)),
        "test_precision_macro": float(precision_macro),
        "test_recall_macro": float(recall_macro),
        "test_f1_macro": float(f1_macro),
        "best_classes_by_f1": [
            {"class": name, "f1": per_class[name]["f1-score"]}
            for name in reversed(ordered_classes[-5:])
        ],
        "worst_classes_by_f1": [
            {"class": name, "f1": per_class[name]["f1-score"]}
            for name in ordered_classes[:5]
        ],
    }
    (reports_dir / "cnn_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "MLP C_grande (EP1)",
                "input_format": "vector 784",
                "parameters": 569240,
                "epochs_completed": 20,
                "fit_seconds": 31.5,
                "validation_accuracy": 1.0,
                "test_accuracy": 0.7971277237,
                "test_f1_macro": 0.7760310432,
            },
            {
                "model": "CNN (EP2)",
                "input_format": "tensor 28x28x1",
                "parameters": model.count_params(),
                "epochs_completed": len(history.history["loss"]),
                "fit_seconds": fit_seconds,
                "validation_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "test_f1_macro": f1_macro,
            },
        ]
    )
    comparison.to_csv(reports_dir / "mlp_cnn_comparison.csv", index=False)
    save_figure(
        plot_model_comparison(comparison),
        images_dir / "mlp_cnn_comparison.png",
    )

    save_figure(plot_history(history), images_dir / "cnn_training_curves.png")
    save_figure(
        plot_confusion(y_test, predictions, index_to_label),
        images_dir / "cnn_confusion_matrix.png",
    )
    save_figure(
        plot_prediction_examples(
            X_test, y_test, predictions, index_to_label, correct=True
        ),
        images_dir / "cnn_correct_examples.png",
    )
    save_figure(
        plot_prediction_examples(
            X_test, y_test, predictions, index_to_label, correct=False
        ),
        images_dir / "cnn_incorrect_examples.png",
    )

    feature_index = int(np.where(y_test == predictions)[0][0])
    save_figure(
        plot_feature_maps(model, X_test[feature_index]),
        images_dir / "cnn_feature_maps.png",
    )

    wrong = np.where(y_test != predictions)[0][:3]
    correct = np.where(y_test == predictions)[0][:3]
    selected = np.concatenate([correct, wrong])
    save_figure(
        plot_gradcam_examples(
            model,
            X_test[selected],
            y_test[selected],
            predictions[selected],
            index_to_label,
            max_items=len(selected),
        ),
        images_dir / "cnn_gradcam_examples.png",
    )

    model.save(models_dir / "cnn_sign_language.keras")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
