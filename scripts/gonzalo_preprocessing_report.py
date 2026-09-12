from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from sign_mlp.data import (  # noqa: E402
    IMAGE_SIZE,
    INPUT_DIM,
    as_images,
    label_to_letter,
    prepare_sign_mnist_data,
    read_sign_mnist_frames,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"


def load_font(size: int):
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_class_count_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    train_counts = train_df["label"].value_counts().sort_index()
    test_counts = test_df["label"].value_counts().sort_index()
    labels = sorted(set(train_counts.index).union(test_counts.index))

    lines = [
        "| Label original | Letra | Train | Test |",
        "| --- | --- | ---: | ---: |",
    ]
    for label in labels:
        lines.append(
            f"| {int(label)} | {label_to_letter(label)} | "
            f"{int(train_counts.get(label, 0))} | {int(test_counts.get(label, 0))} |"
        )
    return "\n".join(lines)


def plot_class_distribution(train_df: pd.DataFrame, output_path: Path) -> None:
    counts = train_df["label"].value_counts().sort_index()
    letters = [label_to_letter(label) for label in counts.index]

    width, height = 1200, 430
    margin_left, margin_right = 70, 30
    margin_top, margin_bottom = 55, 70
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_count = int(counts.max())
    bar_gap = 6
    bar_width = max(8, int((chart_width - bar_gap * (len(counts) - 1)) / len(counts)))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    label_font = load_font(13)
    tick_font = load_font(12)

    draw.text(
        (margin_left, 16),
        "Distribucion de clases en entrenamiento",
        fill="#111111",
        font=title_font,
    )
    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill="#333333",
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill="#333333",
        width=2,
    )

    for step in range(5):
        value = int(max_count * step / 4)
        y = height - margin_bottom - int(chart_height * step / 4)
        draw.line([(margin_left, y), (width - margin_right, y)], fill="#dddddd", width=1)
        draw.text((12, y - 7), str(value), fill="#333333", font=tick_font)

    for index, (letter, count) in enumerate(zip(letters, counts.values)):
        x0 = margin_left + index * (bar_width + bar_gap)
        x1 = x0 + bar_width
        bar_height = int(chart_height * int(count) / max_count)
        y0 = height - margin_bottom - bar_height
        y1 = height - margin_bottom
        draw.rectangle([(x0, y0), (x1, y1)], fill="#276fbf")
        bbox = draw.textbbox((0, 0), letter, font=label_font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x0 + bar_width // 2 - text_width // 2, height - margin_bottom + 14),
            letter,
            fill="#111111",
            font=label_font,
        )

    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    image.save(output_path, optimize=True)


def plot_sample_grid(X_train_full: np.ndarray, y_train_full: np.ndarray, output_path: Path) -> None:
    labels = sorted(np.unique(y_train_full))
    cols = 6
    rows = int(np.ceil(len(labels) / cols))
    cell_width, cell_height = 90, 92
    title_height = 32
    image = Image.new("RGB", (cols * cell_width, title_height + rows * cell_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(18)
    label_font = load_font(14)
    draw.text((18, 10), "Ejemplo de imagen por clase", fill="#111111", font=title_font)

    for index, label in enumerate(labels):
        row = index // cols
        col = index % cols
        x = col * cell_width + 17
        y = title_height + row * cell_height + 8
        first_index = np.where(y_train_full == label)[0][0]
        array = (as_images(X_train_full[[first_index]])[0] * 255).astype(np.uint8)
        sample = Image.fromarray(array, mode="L").resize((56, 56), Image.Resampling.NEAREST)
        image.paste(sample.convert("RGB"), (x, y + 20))
        letter = label_to_letter(label)
        bbox = draw.textbbox((0, 0), letter, font=label_font)
        text_width = bbox[2] - bbox[0]
        draw.text((x + 28 - text_width // 2, y), letter, fill="#111111", font=label_font)

    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=32)
    image.save(output_path, optimize=True)


def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame, prepared: dict) -> str:
    train_pixels = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    test_pixels = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    raw_pixels = np.concatenate([train_pixels.reshape(-1), test_pixels.reshape(-1)])

    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_val = prepared["y_val"]
    y_test = prepared["y_test"]
    metadata = prepared["metadata"]

    class_list = ", ".join(metadata["class_names"])
    class_count_table = make_class_count_table(train_df, test_df)

    return f"""# Avance Gonzalo Castillo: carga y preprocesamiento

## Que hice

- Tome los CSV originales del dataset Sign Language MNIST desde `data/raw/`.
- Separe la columna `label` de los pixeles de cada imagen.
- Normalice los pixeles desde el rango 0-255 al rango 0-1.
- Prepare la entrada del MLP como vectores de {INPUT_DIM} valores por imagen ({IMAGE_SIZE}x{IMAGE_SIZE}).
- Cree una division `train/validation` estratificada desde el set de entrenamiento oficial.
- Deje el set de test oficial separado para la evaluacion final.

## Resumen del dataset

- Dataset: Sign Language MNIST.
- Tipo de problema: clasificacion multiclase.
- Clases usadas: {metadata["num_classes"]}.
- Letras presentes: {class_list}.
- Formato original: imagenes en escala de grises de {IMAGE_SIZE}x{IMAGE_SIZE} pixeles.
- Formato para el MLP: vector plano de {INPUT_DIM} pixeles.

## Particiones preparadas

| Particion | Filas | Columnas de entrada | Labels |
| --- | ---: | ---: | ---: |
| Train | {X_train.shape[0]} | {X_train.shape[1]} | {len(y_train)} |
| Validation | {X_val.shape[0]} | {X_val.shape[1]} | {len(y_val)} |
| Test | {X_test.shape[0]} | {X_test.shape[1]} | {len(y_test)} |

## Normalizacion

| Etapa | Minimo | Maximo |
| --- | ---: | ---: |
| Pixeles originales | {raw_pixels.min():.0f} | {raw_pixels.max():.0f} |
| Pixeles normalizados | {prepared["X_train_full"].min():.4f} | {prepared["X_train_full"].max():.4f} |

## Distribucion de clases

{class_count_table}

## Evidencia generada

- `images/gonzalo_class_distribution.png`: distribucion de clases de entrenamiento.
- `images/gonzalo_sample_grid.png`: ejemplo visual de una imagen por clase.

## Nota para la defensa

Esta parte prepara los datos para que el MLP pueda entrenar. El punto clave es que una imagen
de {IMAGE_SIZE}x{IMAGE_SIZE} se aplana a {INPUT_DIM} entradas numericas; eso permite usar capas
densas, pero tambien hace que el modelo pierda parte de la estructura espacial de la imagen.
"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = read_sign_mnist_frames(RAW_DIR)
    prepared = prepare_sign_mnist_data(RAW_DIR, validation_size=0.2, random_state=42)

    plot_class_distribution(train_df, IMAGES_DIR / "gonzalo_class_distribution.png")
    plot_sample_grid(
        prepared["X_train_full"],
        prepared["y_train_full"],
        IMAGES_DIR / "gonzalo_sample_grid.png",
    )

    report = build_report(train_df, test_df, prepared)
    report_path = REPORTS_DIR / "gonzalo_preprocessing_summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Generated {report_path}")
    print(f"Generated {IMAGES_DIR / 'gonzalo_class_distribution.png'}")
    print(f"Generated {IMAGES_DIR / 'gonzalo_sample_grid.png'}")


if __name__ == "__main__":
    main()
