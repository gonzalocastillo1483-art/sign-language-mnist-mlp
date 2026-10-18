"""Build the documented and executable EP2 CNN notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "03_sign_language_mnist_cnn_ep2.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# Evaluación Parcial 2: Sign Language MNIST con CNN

Este notebook reutiliza las 24 clases estáticas de Sign Language MNIST de la EP1 y conserva la misma partición estratificada con semilla 42. La variante del grupo compara una CNN con el MLP `C_grande` bajo los mismos datos de entrenamiento, validación y test.

La CNN conserva cada imagen como un tensor de forma `28 x 28 x 1`. Esto permite que las capas convolucionales aprendan patrones locales, a diferencia del MLP, que recibe un vector plano de 784 valores.
"""
    ),
    md(
        """
## 1. Problema, objetivos y KPIs

El objetivo es clasificar una imagen estática de una seña ASL en una de las 24 letras disponibles. Se evaluarán accuracy, precision macro, recall macro, F1 macro y matriz de confusión. La comparación con la EP1 también considera cantidad de parámetros, tiempo de entrenamiento y diferencia entre validación y test.
"""
    ),
    code(
        """
from pathlib import Path
import json
import sys
import time

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

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

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
set_global_seed(SEED)
"""
    ),
    md(
        """
## 2. Preparación de datos para CNN

Los píxeles se convierten a `float32` y se dividen por 255. La diferencia frente a la EP1 está en el formato: el MLP utiliza `(n, 784)` y la CNN utiliza `(n, 28, 28, 1)`. El último valor representa el canal de escala de grises.
"""
    ),
    code(
        """
data = prepare_sign_mnist_cnn_data(
    ROOT / "data" / "raw",
    validation_size=0.2,
    random_state=SEED,
)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
index_to_label = data["index_to_label"]
class_names = data["metadata"]["class_names"]

print("Train:", X_train.shape, y_train.shape)
print("Validación:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)
print("Rango normalizado:", float(X_train.min()), float(X_train.max()))
print("Clases:", class_names)
"""
    ),
    code(
        """
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for ax, image, label in zip(axes.flat, X_train[:12], y_train[:12]):
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(class_names[int(label)])
    ax.axis("off")
fig.suptitle("Ejemplos del conjunto de entrenamiento")
fig.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 3. Diseño de la CNN

La arquitectura utiliza tres bloques convolucionales con filtros `3 x 3`. Las capas aumentan de 32 a 128 filtros para aprender representaciones progresivamente más específicas. `MaxPooling2D` reduce el tamaño espacial y conserva las activaciones dominantes. Dropout reduce la dependencia de neuronas específicas. La capa softmax entrega una distribución sobre 24 clases.
"""
    ),
    code(
        """
model = build_cnn(
    input_shape=(28, 28, 1),
    num_classes=24,
    learning_rate=LEARNING_RATE,
    conv_dropout=0.25,
    dense_dropout=0.4,
)
model.summary()
"""
    ),
    md(
        """
## 4. Entrenamiento

Se mantiene un máximo de 20 épocas y batch 128 para conservar la trazabilidad con la EP1. Adam comienza con learning rate 0,001. `ReduceLROnPlateau` reduce el paso cuando la pérdida de validación se estanca y `EarlyStopping` conserva los mejores pesos según `val_loss`.
"""
    ),
    code(
        """
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
print(f"Tiempo de entrenamiento: {fit_seconds:.2f} segundos")
"""
    ),
    code(
        """
plot_history(history)
plt.show()
"""
    ),
    md(
        """
## 5. Evaluación de test

El test oficial se consulta después de fijar la arquitectura. Accuracy resume aciertos globales. Las métricas macro asignan el mismo peso a cada letra, aunque tengan cantidades distintas de ejemplos.
"""
    ),
    code(
        """
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
predictions = np.argmax(probabilities, axis=1)

precision_macro = precision_score(y_test, predictions, average="macro", zero_division=0)
recall_macro = recall_score(y_test, predictions, average="macro", zero_division=0)
f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

metrics = pd.Series({
    "accuracy_test": test_accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "loss_test": test_loss,
})
metrics
"""
    ),
    code(
        """
report_text = classification_report(
    y_test,
    predictions,
    target_names=class_names,
    zero_division=0,
)
print(report_text)
"""
    ),
    code(
        """
plot_confusion(y_test, predictions, index_to_label)
plt.show()
"""
    ),
    code(
        """
plot_prediction_examples(X_test, y_test, predictions, index_to_label, correct=True)
plt.show()
plot_prediction_examples(X_test, y_test, predictions, index_to_label, correct=False)
plt.show()
"""
    ),
    md(
        """
## 6. Feature maps y mapas de activación

Las primeras capas mantienen bordes, contrastes y trazos locales. Las capas profundas producen mapas más pequeños y selectivos. Grad-CAM resume qué regiones de la última capa convolucional aportaron a una clase predicha. Estos mapas orientan la interpretación, pero no prueban por sí solos una causa del acierto o error.
"""
    ),
    code(
        """
feature_index = int(np.where(y_test == predictions)[0][0])
plot_feature_maps(model, X_test[feature_index])
plt.show()
"""
    ),
    code(
        """
correct_indices = np.where(y_test == predictions)[0][:3]
wrong_indices = np.where(y_test != predictions)[0][:3]
selected = np.concatenate([correct_indices, wrong_indices])
plot_gradcam_examples(
    model,
    X_test[selected],
    y_test[selected],
    predictions[selected],
    index_to_label,
    max_items=len(selected),
)
plt.show()
"""
    ),
    md(
        """
## 7. Comparación experimental MLP y CNN

La referencia del MLP corresponde a la ejecución final de la EP1. Ambos modelos usan las mismas 24 clases y el mismo test oficial. La CNN tarda algo más en CPU, pero aprovecha la vecindad espacial y necesita menos parámetros que el MLP grande.
"""
    ),
    code(
        """
comparison = pd.DataFrame([
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
])
comparison
"""
    ),
    code(
        """
plot_data = comparison.set_index("model")[["test_accuracy", "test_f1_macro"]] * 100
ax = plot_data.plot(kind="bar", figsize=(9, 5), ylim=(0, 105))
ax.set_ylabel("Porcentaje")
ax.set_title("Comparación MLP y CNN en test")
ax.tick_params(axis="x", rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=3)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
## 8. Conclusiones

La CNN mejora la generalización en el test porque sus filtros comparten pesos y buscan patrones locales en la imagen. El resultado no demuestra funcionamiento en cámaras, personas nuevas o señas en movimiento. Las clases con menor F1 y los ejemplos incorrectos deben revisarse antes de pensar en un uso educativo automático.

Como mejoras futuras se pueden comparar aumentación de datos, regularización L2 y otras profundidades, cambiando una variable a la vez y reservando una evaluación independiente.
"""
    ),
    code(
        """
reports_dir = ROOT / "reports"
images_dir = ROOT / "images"
models_dir = ROOT / "models"
for directory in (reports_dir, images_dir, models_dir):
    directory.mkdir(parents=True, exist_ok=True)

pd.DataFrame(history.history).to_csv(reports_dir / "cnn_history.csv", index_label="epoch")
comparison.to_csv(reports_dir / "mlp_cnn_comparison.csv", index=False)
(reports_dir / "cnn_classification_report.txt").write_text(report_text, encoding="utf-8")
notebook_metrics = {
    "model": "CNN_EP2",
    "seed": SEED,
    "parameters": int(model.count_params()),
    "fit_seconds": float(fit_seconds),
    "validation_loss": float(val_loss),
    "validation_accuracy": float(val_accuracy),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_precision_macro": float(precision_macro),
    "test_recall_macro": float(recall_macro),
    "test_f1_macro": float(f1_macro),
}
(reports_dir / "cnn_metrics_notebook.json").write_text(
    json.dumps(notebook_metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
model.save(models_dir / "cnn_sign_language.keras")
print("Evidencias actualizadas en reports/, images/ y models/.")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
