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


def feature_map_model(model, layer_names=("conv_1", "conv_2", "conv_3")):
    """Create a model that exposes intermediate convolutional activations."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements-ep2.txt or use Colab."
        ) from exc

    outputs = [model.get_layer(name).output for name in layer_names]
    return keras.Model(inputs=model.inputs, outputs=outputs, name="feature_maps")


def plot_feature_maps(model, image, layer_names=("conv_1", "conv_2", "conv_3"), maps_per_layer=8):
    """Plot representative feature maps from selected convolutional layers."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    activations = feature_map_model(model, layer_names).predict(image, verbose=0)
    fig, axes = plt.subplots(
        len(layer_names),
        maps_per_layer,
        figsize=(2 * maps_per_layer, 2.2 * len(layer_names)),
    )
    axes = np.atleast_2d(axes)

    for row, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        available = min(maps_per_layer, activation.shape[-1])
        for col in range(maps_per_layer):
            ax = axes[row, col]
            ax.axis("off")
            if col < available:
                ax.imshow(activation[0, :, :, col], cmap="viridis")
            if col == 0:
                ax.set_title(layer_name)

    fig.suptitle("Mapas de caracteristicas por profundidad")
    fig.tight_layout()
    return fig


def gradcam_heatmap(model, image, class_index=None, conv_layer_name="conv_3"):
    """Compute a Grad-CAM heatmap for one image and one predicted or chosen class."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements-ep2.txt or use Colab."
        ) from exc

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = keras.Model(
        model.inputs,
        [conv_layer.output, model.outputs[0]],
        name="gradcam",
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    weighted = conv_output[0] * pooled_gradients
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy(), int(class_index)


def plot_gradcam_examples(model, X, y_true, y_pred, index_to_label, max_items=6):
    """Plot test images beside Grad-CAM maps for their predicted classes."""
    count = min(max_items, len(X))
    fig, axes = plt.subplots(count, 2, figsize=(6, 2.6 * count))
    axes = np.atleast_2d(axes)

    for row in range(count):
        heatmap, predicted_index = gradcam_heatmap(
            model,
            X[row],
            class_index=int(y_pred[row]),
        )
        real = label_to_letter(index_to_label[int(y_true[row])])
        predicted = label_to_letter(index_to_label[predicted_index])

        axes[row, 0].imshow(X[row].squeeze(), cmap="gray")
        axes[row, 0].set_title(f"Real: {real} | Pred: {predicted}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="inferno")
        axes[row, 1].set_title("Activacion Grad-CAM")
        axes[row, 1].axis("off")

    fig.tight_layout()
    return fig
