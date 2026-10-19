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

## Resultados de la ejecución de Jason

El cierre usa el notebook completo ejecutado con semilla 42 y las particiones originales. Sus evidencias están en `output/jason_ep2/`.

| Métrica | MLP EP1 | CNN Jason EP2 |
| --- | ---: | ---: |
| Parámetros | 569.240 | 243.352 |
| Tiempo registrado | 31,45 s | 137,20 s |
| Accuracy de validación | 100,00 % | 100,00 % |
| Accuracy de test | 79,71 % | **98,03 %** |
| F1 macro de test | 0,7760 | **0,9786** |

La CNN mejora el accuracy en **18,32 puntos porcentuales** y el F1 macro en **0,2025**, con **57,25 % menos parámetros**. La brecha validación-test baja de 20,29 a **1,97 puntos**. Los tiempos corresponden a entornos distintos y no constituyen una comparación controlada de velocidad.

Las clases con menor F1 son T, R y X. Las principales confusiones verificadas desde el modelo guardado son T como X (22), R como V (21) y X como T (21). Las ejecuciones previas de 96,60 % y 96,81 % se conservan como antecedentes, sin combinar sus métricas con la corrida actual.

## Division del trabajo EP2

El trabajo se divide por bloques completos para mantener una carga equilibrada.
Cada integrante es responsable de una parte tecnica, su evidencia, la redaccion
correspondiente y su explicacion durante la presentacion.

### Gonzalo Castillo: continuidad y preparacion para la CNN

- Verificar la reutilizacion del dataset, las clases, los splits y la semilla 42.
- Revisar la normalizacion de pixeles y la conversion de vectores de 784 valores
  a tensores `(28, 28, 1)`.
- Mantener documentado el flujo de datos y la reproducibilidad del proyecto.
- Redactar las secciones 1, 2 y 3 del informe tecnico.
- Mantener actualizado este README.
- Preparar y presentar las diapositivas 1, 2 y 3.

### Jenaro Marin: arquitectura, entrenamiento e interpretabilidad

- Implementar y justificar las capas convolucionales, pooling, dropout y densas.
- Configurar y explicar Adam, la funcion de perdida y los callbacks.
- Ejecutar el entrenamiento y revisar las curvas de accuracy y perdida.
- Generar e interpretar Feature Maps y Grad-CAM.
- Redactar las secciones 4 y 5 del informe tecnico.
- Preparar y presentar las diapositivas 4, 5 y 6.

### Jason Aguilera: evaluacion, comparacion y conclusiones

- Calcular e interpretar accuracy, precision, recall y F1 macro.
- Analizar la matriz de confusion y ejemplos correctos e incorrectos.
- Comparar objetivamente la CNN con el MLP de la EP1.
- Explicar limitaciones, mejoras posibles y conclusiones.
- Redactar las secciones 6, 7, 8 y 9 del informe tecnico.
- Preparar y presentar las diapositivas 7, 8, 9 y 10.

### Responsabilidades compartidas

- Ejecutar el notebook completo antes de entregar.
- Verificar que README, informe, notebook y PPT utilicen los mismos resultados.
- Ensayar la exposicion completa con cronometro hasta llegar a 10 minutos.
- Conocer el flujo completo, porque el docente puede hacer preguntas cruzadas.
- Hacer al menos una pregunta durante la presentacion de otro grupo.

Para evitar conflictos, antes de comenzar cada integrante debe ejecutar
`git pull origin ep2-cnn` y subir su bloque con commits descriptivos en espanol.
El notebook, el informe y la PPT no deben editarse simultaneamente.

La asignacion detallada y las tareas compartidas estan en
[`presentation/reparto_ep2.md`](presentation/reparto_ep2.md).

## Entregables EP2

- [Notebook ejecutable CNN](notebooks/03_sign_language_mnist_cnn_ep2.ipynb).
- [Informe tecnico EP2](INFORME_TECNICO_EP2.md).
- [Presentación EP2 integrada](presentation/archivos/Sign_Language_MNIST_CNN_EP2_Jason.pptx).
- [Guion de presentacion y defensa](presentation/guion_ep2.md).
- [Guia individual de Jenaro para la EP2](presentation/jenaro_defensa_ep2.md).
- [Métricas de la ejecución elegida](output/jason_ep2/reports/cnn_metrics.json).
- [Reporte por clase](output/jason_ep2/reports/cnn_classification_report.txt).
- [Curvas de entrenamiento](output/jason_ep2/images/cnn_training_curves.png).
- [Matriz de confusion](output/jason_ep2/images/cnn_confusion_matrix.png).
- [Mapas de caracteristicas](output/jason_ep2/images/cnn_feature_maps.png).
- [Ejemplos Grad-CAM](output/jason_ep2/images/cnn_gradcam_examples.png).
- [Predicciones correctas](output/jason_ep2/images/cnn_correct_examples.png) e
  [incorrectas](output/jason_ep2/images/cnn_incorrect_examples.png).

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

## Continuar desde el aporte de Jenaro EP2

Para incorporar el bloque de arquitectura, entrenamiento e interpretabilidad:

```powershell
git switch ep2-cnn
git pull --ff-only origin ep2-cnn
```

La [revision de Jenaro](reports/jenaro_ep2_revision.md) detalla los cambios y
comprobaciones. El [notebook ejecutado](output/jenaro_ep2/03_sign_language_mnist_cnn_ep2_ejecutado.ipynb),
los reportes y las figuras de su corrida estan en `output/jenaro_ep2/`.
La [PPT con las diapositivas 4 a 6 revisadas](presentation/archivos/Sign_Language_MNIST_CNN_EP2_Jenaro.pptx)
conserva las diez diapositivas y las cifras de referencia del equipo.

La referencia histórica obtuvo 96,60 % de test y 41,41 s. La corrida de Jenaro
obtuvo 96,81 %, F1 macro 0,9630 y 127,95 s en otro entorno. El cierre actual
utiliza la ejecución de Jason: 98,03 %, F1 macro 0,9786 y 137,20 s. Las corridas
se mantienen identificadas por separado. Los tiempos entre entornos distintos
no permiten comparar velocidad.

El codigo comparte la exportacion completa entre notebook y consola e incluye
Grad-CAM de las 24 clases. El modelo `.keras` se reconstruye localmente:

```powershell
python scripts\train_cnn_ep2.py --output-dir output\jenaro_ep2
```

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

## Evaluación y cierre de Jason

- [Notebook completo ejecutado](output/jason_ep2/03_sign_language_mnist_cnn_ep2_ejecutado.ipynb).
- [Guion individual](presentation/GUION_JASON_EP2.md) y [diapositivas 7 a 10](presentation/archivos/Jason_EP2_Diapositivas_7_a_10.pptx).
- [Evaluación verificada](output/jason_ep2/reports/jason_evaluacion_verificada.json), [predicciones](output/jason_ep2/reports/cnn_predictions.csv) y [matriz numérica](output/jason_ep2/reports/cnn_confusion_matrix.csv).
- [Comparación con métricas y tiempo original del MLP](output/jason_ep2/reports/mlp_cnn_comparison_exact.csv).

El script `scripts/evaluar_jason_ep2.py` vuelve a predecir desde los pesos guardados, comprueba las métricas y exporta evidencia exacta sin entrenar. Ejecutar desde la raíz:

```powershell
.\.venv\Scripts\python.exe scripts\evaluar_jason_ep2.py --project-root .
```

El modelo permanece local en `output/jason_ep2/models/cnn_sign_language.keras` y está excluido de Git. El cierre conserva los aportes de arquitectura e interpretabilidad de Jenaro. La sección 5 solo actualiza la corrida y sus evidencias para coincidir con la evaluación de las secciones 6 a 9.
