import random

import numpy as np


def set_global_seed(seed=42):
    """Set Python, NumPy, and TensorFlow seeds for reproducible runs."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements-ep2.txt or use Colab."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def build_cnn(
    input_shape=(28, 28, 1),
    num_classes=24,
    learning_rate=0.001,
    conv_dropout=0.25,
    dense_dropout=0.4,
):
    """Build the EP2 CNN for 24-class Sign Language MNIST classification."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements-ep2.txt or use Colab."
        ) from exc

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape, name="imagen_28x28"),
            layers.Conv2D(
                32,
                kernel_size=3,
                padding="same",
                activation="relu",
                name="conv_1",
            ),
            layers.MaxPooling2D(pool_size=2, name="pool_1"),
            layers.Conv2D(
                64,
                kernel_size=3,
                padding="same",
                activation="relu",
                name="conv_2",
            ),
            layers.MaxPooling2D(pool_size=2, name="pool_2"),
            layers.Dropout(conv_dropout, name="dropout_conv"),
            layers.Conv2D(
                128,
                kernel_size=3,
                padding="same",
                activation="relu",
                name="conv_3",
            ),
            layers.MaxPooling2D(pool_size=2, name="pool_3"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_128"),
            layers.Dropout(dense_dropout, name="dropout_dense"),
            layers.Dense(num_classes, activation="softmax", name="salida_softmax"),
        ],
        name="sign_language_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def training_callbacks(patience=4):
    """Return callbacks that keep the best validation weights and adjust learning."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements-ep2.txt or use Colab."
        ) from exc

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
