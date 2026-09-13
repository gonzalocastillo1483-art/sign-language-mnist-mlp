def build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_layers=(256, 128),
    dropout=0.25,
    learning_rate=0.001,
    loss="sparse_categorical_crossentropy",
):
    """Construye un MLP con capas Dense, ReLU y salida softmax.

    Cada neurona calcula una suma ponderada más un sesgo y aplica ReLU.
    La salida tiene una neurona por clase. Adam actualiza los pesos para
    reducir la crossentropy.

    Con etiquetas enteras usar sparse_categorical_crossentropy (valor
    original del proyecto). Con one-hot, como en la actividad 1.4.4,
    usar categorical_crossentropy. Dropout=0 deja solo capas densas.
    """
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
        loss=loss,
        metrics=["accuracy"],
    )
    return model
