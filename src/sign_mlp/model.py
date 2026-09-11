def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_layers=(256, 128),
    dropout=0.25,
    learning_rate=0.001,
):
    """Build a Keras MLP for image classification."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install requirements.txt or run the notebook in Colab."
        ) from exc

    model = keras.Sequential(name="sign_language_mlp")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu"))
        if dropout:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
