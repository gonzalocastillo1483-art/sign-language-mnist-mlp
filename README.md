# Clasificacion de lenguaje de senas: MLP y CNN

Proyecto del equipo formado por **Gonzalo Castillo, Jenaro Marin y Jason
Aguilera** para las evaluaciones EP1 y EP2 de TLY1102.

- **EP1:** modelo base MLP para clasificacion multiclase.
- **EP2:** red neuronal convolucional (CNN) sobre el mismo problema y dataset.

La rama `main` conserva la entrega MLP. La rama `ep2-cnn` desarrolla la CNN y
la comparacion objetiva entre ambos modelos.

## Problema y dataset

El objetivo es clasificar imagenes de manos que representan letras del
alfabeto ASL. Se utiliza **Sign Language MNIST**:

- 27.455 imagenes de entrenamiento y 7.172 de test oficial.
- Imagenes de 28 x 28 pixeles en escala de grises.
- 24 clases: A-I y K-Y.
- J y Z no aparecen porque esas senas requieren movimiento.
- Cada CSV contiene una columna `label` y 784 columnas de pixeles.

Se mantiene el mismo split estratificado de la EP1: 21.964 ejemplos para
entrenamiento, 5.491 para validacion y 7.172 para test oficial.

## Continuidad entre EP1 y EP2

La variante es la misma clasificacion multiclase. Lo que cambia es la forma en
que el modelo recibe y procesa las imagenes:

| Modelo | Entrada | Idea principal |
| --- | --- | --- |
| MLP (EP1) | Vector de 784 valores | Trata cada pixel como una caracteristica independiente. |
| CNN (EP2) | Tensor de 28 x 28 x 1 | Conserva la posicion espacial y aprende patrones locales. |

Para la CNN, los pixeles se convierten de enteros entre 0 y 255 a valores
decimales entre 0 y 1, y cada fila vuelve a su forma de imagen `(28, 28, 1)`.
No se aplica data augmentation porque no fue necesaria para obtener una mejora
clara; queda como alternativa para estudiar mayor robustez.

## Arquitectura CNN

La red implementada usa:

1. `Conv2D(32, 3x3)` y `MaxPooling2D(2x2)`.
2. `Conv2D(64, 3x3)` y `MaxPooling2D(2x2)`.
3. `Dropout(0.25)`.
4. `Conv2D(128, 3x3)` y `MaxPooling2D(2x2)`.
5. `Flatten`, `Dense(128, ReLU)` y `Dropout(0.40)`.
6. `Dense(24, softmax)` para obtener una probabilidad por clase.

El modelo se entrena con Adam (`learning_rate=0.001`),
`sparse_categorical_crossentropy`, batch de 128 y hasta 20 epocas. Incluye
`EarlyStopping` y `ReduceLROnPlateau` para controlar el entrenamiento.

## Resultados de referencia

Ejecucion local con semilla 42:

| Metrica | MLP EP1 | CNN EP2 |
| --- | ---: | ---: |
| Parametros | 569.240 | 243.352 |
| Tiempo de entrenamiento | 31,50 s | 41,41 s |
| Accuracy validacion | 100,00% | 100,00% |
| Accuracy test | 79,71% | **96,60%** |
| F1 macro test | 0,7760 | **0,9595** |

La CNN aumenta el accuracy de test en **16,89 puntos porcentuales**, mejora el
F1 macro en **0,1835** y utiliza aproximadamente **57,25% menos parametros**.
En esta ejecucion tarda cerca de 9,91 segundos mas. La diferencia entre
validacion y test baja de 20,29 puntos en el MLP a 3,40 puntos en la CNN, lo que
indica una generalizacion mucho mejor.

Las clases con menor F1 en la CNN fueron T, X, R, Q y L. Las confusiones mas
frecuentes fueron T con X, G con Q y X con T, senas que comparten formas o
posiciones visuales parecidas.

## Division del trabajo EP2

| Responsable | Parte principal |
| --- | --- |
| Gonzalo Castillo | Continuidad con EP1, justificacion de la variante y adaptacion de vector MLP a tensor CNN. |
| Jenaro Marin | Diseno de la CNN, entrenamiento, curvas, mapas de caracteristicas y Grad-CAM. |
| Jason Aguilera | Metricas, comparacion MLP-CNN, matriz de confusion, errores, limitaciones y conclusiones. |

La asignacion detallada y las tareas compartidas estan en
[`presentation/reparto_ep2.md`](presentation/reparto_ep2.md).

## Entregables EP2

- [Notebook ejecutable CNN](notebooks/03_sign_language_mnist_cnn_ep2.ipynb).
- [Informe tecnico EP2](INFORME_TECNICO_EP2.md).
- [Presentacion EP2](presentation/archivos/Sign_Language_MNIST_CNN_EP2.pptx).
- [Guion de presentacion y defensa](presentation/guion_ep2.md).
- [Metricas en JSON](reports/cnn_metrics.json).
- [Reporte por clase](reports/cnn_classification_report.txt).
- [Curvas de entrenamiento](images/cnn_training_curves.png).
- [Matriz de confusion](images/cnn_confusion_matrix.png).
- [Mapas de caracteristicas](images/cnn_feature_maps.png).
- [Ejemplos Grad-CAM](images/cnn_gradcam_examples.png).
- [Predicciones correctas](images/cnn_correct_examples.png) e
  [incorrectas](images/cnn_incorrect_examples.png).

El modelo entrenado se genera localmente como `models/cnn_sign_language.keras`
y no se versiona porque puede reconstruirse ejecutando el notebook.

## Instalacion y ejecucion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ep2.txt
```

Luego se puede abrir y ejecutar de principio a fin:

```text
notebooks/03_sign_language_mnist_cnn_ep2.ipynb
```

O ejecutar el flujo desde consola:

```powershell
python scripts\train_cnn_ep2.py
```

Los CSV originales ya estan incluidos en `data/raw/`. La ejecucion vuelve a
generar las metricas, figuras y el modelo.

## Estructura

```text
data/raw/              CSV de Sign Language MNIST
images/                Graficos y evidencia visual
models/                Modelos generados localmente
notebooks/             Flujo MLP de EP1 y flujo CNN de EP2
presentation/          PPT, guion y reparto
reports/               Metricas, tablas y reportes
scripts/               Entrenamiento y generacion del notebook
src/sign_mlp/           Codigo reutilizable para datos, modelos y evaluacion
```

## Limitaciones y trabajo futuro

- El test oficial puede tener diferencias de distribucion respecto del split
  interno de validacion.
- Algunas senas estaticas son visualmente muy similares.
- No se reconocen J ni Z ni secuencias en movimiento.
- Como trabajo futuro se puede probar data augmentation, Batch Normalization,
  ajuste de hiperparametros y validacion con imagenes capturadas fuera del
  dataset.

El informe original de la EP1 permanece en
[`INFORME_TECNICO.md`](INFORME_TECNICO.md).
