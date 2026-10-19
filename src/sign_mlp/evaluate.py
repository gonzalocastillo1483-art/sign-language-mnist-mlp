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
    """Show the source image and the first channels of each selected layer."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    activations = feature_map_model(model, layer_names).predict(image, verbose=0)
    if not isinstance(activations, (list, tuple)):
        activations = [activations]
    fig, axes = plt.subplots(
        len(layer_names),
        maps_per_layer + 1,
        figsize=(2 * (maps_per_layer + 1), 2.5 * len(layer_names)),
        squeeze=False,
    )

    for row, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        axes[row, 0].imshow(image[0].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Entrada | {layer_name}\n{activation.shape[1:3]}")
        axes[row, 0].axis("off")
        available = min(maps_per_layer, activation.shape[-1])
        for col in range(maps_per_layer):
            ax = axes[row, col + 1]
            ax.axis("off")
            if col < available:
                ax.imshow(activation[0, :, :, col], cmap="viridis")
                ax.set_title(f"Canal {col}")

    fig.suptitle("Mapas por profundidad: primeros 8 canales, escala por mapa")
    fig.tight_layout()
    return fig


def gradcam_heatmap(model, image, class_index=None, conv_layer_name="conv_3"):
    """Explain a class logit, without changing the model's softmax or weights.

    The EP2 classifier ends in a Dense softmax layer. Reconstruct its logits
    from the layer input to avoid saturated softmax gradients. Algorithm:
    https://keras.io/examples/vision/grad_cam/
    """
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
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects exactly one image, with its channel axis.")

    if not isinstance(model, keras.Sequential):
        raise ValueError("Grad-CAM expects the EP2 Sequential CNN.")
    model.get_layer(conv_layer_name)
    output_layer = model.layers[-1]
    if not isinstance(output_layer, keras.layers.Dense):
        raise ValueError("Grad-CAM requires a final Dense classifier.")
    with tf.GradientTape() as tape:
        # Reuse the actual layers in order. Serialized Sequential models can
        # retain multiple symbolic nodes; connecting their .output tensors
        # may expose a different node from the one used by the classifier.
        features = tf.convert_to_tensor(image)
        conv_output = None
        for layer in model.layers[:-1]:
            features = layer(features, training=False)
            if layer.name == conv_layer_name:
                conv_output = features
                tape.watch(conv_output)
        if conv_output is None:
            raise ValueError("The selected convolution must precede the classifier.")
        logits = tf.linalg.matmul(features, output_layer.kernel)
        if output_layer.use_bias:
            logits = logits + output_layer.bias
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        class_index = int(class_index)
        if not 0 <= class_index < output_layer.units:
            raise ValueError("class_index is outside the classifier output range.")
        class_score = logits[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise ValueError("The selected convolution is not connected to the class score.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    weighted = conv_output[0] * pooled_gradients
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, maximum)
    return heatmap.numpy(), int(class_index)


def _show_gradcam_overlay(ax, image, heatmap):
    """Resize a normalized map to the image coordinates and overlay it."""
    import tensorflow as tf

    image = np.asarray(image).squeeze()
    resized = tf.image.resize(
        heatmap[..., np.newaxis], image.shape, method="bilinear"
    ).numpy().squeeze()
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(resized, cmap="inferno", vmin=0, vmax=1, alpha=0.65 * resized)
    ax.axis("off")


def plot_gradcam_examples(model, X, y_true, y_pred, index_to_label, max_items=6):
    """Plot original images beside overlays explaining their predicted classes."""
    count = min(max_items, len(X))
    if count < 1:
        raise ValueError("At least one example is required for Grad-CAM.")
    fig, axes = plt.subplots(count, 2, figsize=(6, 2.6 * count), squeeze=False)

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

        _show_gradcam_overlay(axes[row, 1], X[row], heatmap)
        axes[row, 1].set_title(f"Grad-CAM de {predicted}")

    fig.suptitle("Clase predicha | mapa 7x7 ampliado, intensidad relativa por imagen")
    fig.tight_layout()
    return fig


def select_class_examples(y_true, y_pred, index_to_label):
    """Choose the first correct example per true class, or its first error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    selected = []
    for class_index in sorted(index_to_label):
        candidates = np.flatnonzero(y_true == class_index)
        if not len(candidates):
            raise ValueError(f"No examples available for class {class_index}.")
        correct = candidates[y_pred[candidates] == class_index]
        selected.append(int(correct[0] if len(correct) else candidates[0]))
    return selected


def plot_gradcam_by_class(model, X, y_true, y_pred, index_to_label):
    """Show a true-class Grad-CAM for every class, including misclassified ones."""
    selected = select_class_examples(y_true, y_pred, index_to_label)
    groups_per_row = 4
    rows = int(np.ceil(len(selected) / groups_per_row))
    fig, axes = plt.subplots(
        rows, groups_per_row * 2, figsize=(18, rows * 2.6), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for position, (class_index, sample_index) in enumerate(
        zip(sorted(index_to_label), selected)
    ):
        row, group = divmod(position, groups_per_row)
        original, overlay = axes[row, group * 2:group * 2 + 2]
        real = label_to_letter(index_to_label[class_index])
        predicted = label_to_letter(index_to_label[int(y_pred[sample_index])])
        heatmap, _ = gradcam_heatmap(model, X[sample_index], class_index=class_index)
        original.imshow(X[sample_index].squeeze(), cmap="gray", vmin=0, vmax=1)
        original.set_title(f"Real {real} | pred {predicted}\nTest #{sample_index}", fontsize=10)
        _show_gradcam_overlay(overlay, X[sample_index], heatmap)
        overlay.set_title(f"Grad-CAM clase {real}", fontsize=10)
    fig.suptitle(
        "Activacion de las 24 clases ASL | una imagen por clase real\n"
        "Primer acierto disponible; si no existe, primer error. "
        "Mapa 7x7 ampliado, intensidad relativa por imagen."
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.5, w_pad=0.8)
    return fig
