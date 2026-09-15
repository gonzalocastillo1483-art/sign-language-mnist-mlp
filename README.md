# Clasificacion de lenguaje de senas con MLP

Proyecto para la Evaluacion Parcial 1 de TLY1102. El objetivo es construir,
entrenar y defender un modelo de red neuronal tipo MLP para clasificar imagenes
del dataset Sign Language MNIST.

## Decision del equipo

Dataset elegido: `Sign Language MNIST.zip`.

La variante trabajada corresponde a un problema de clasificacion multiclase:
clasificar las letras disponibles del alfabeto ASL usando imagenes de 28x28
pixeles en escala de grises. El dataset contiene 24 clases: A-I y K-Y. No
incluye J ni Z porque esas senas requieren movimiento y el dataset contiene
imagenes estaticas.

## Problema de negocio

Una aplicacion educativa o de accesibilidad necesita reconocer senas de letras
del alfabeto a partir de imagenes. El modelo propuesto busca clasificar cada
imagen en la letra correspondiente, como primera version de un sistema de apoyo
para aprendizaje de lenguaje de senas.

## Objetivos

- Implementar un flujo completo de ciencia de datos para clasificacion de
  imagenes.
- Entrenar una red neuronal MLP como modelo base.
- Evaluar el desempeno con metricas de clasificacion.
- Analizar errores y limitaciones del modelo.
- Documentar el proceso de forma reproducible.

## KPIs

- Accuracy de validacion y test.
- F1-score macro para medir desempeno balanceado entre clases.
- Precision y recall por clase.
- Diferencia entre accuracy de validacion y test para observar generalizacion.
- Letras con mayor confusion en la matriz de confusion.

## Estado actual

| Responsable | Aporte principal | Estado |
| --- | --- | --- |
| Gonzalo Castillo | Carga, normalizacion y particion de datos | Completo |
| Jenaro | Diseno, entrenamiento y comparacion de modelos MLP | Completo |
| Jason | Evaluacion final, matriz de confusion, ejemplos y conclusiones | Completo |

El flujo completo ya esta implementado en
[`notebooks/02_sign_language_mnist_mlp_jenaro.ipynb`](notebooks/02_sign_language_mnist_mlp_jenaro.ipynb).
El notebook 01 se conserva como version inicial del equipo.

## Entregables

Repositorio: https://github.com/gonzalocastillo1483-art/sign-language-mnist-mlp

Para la entrega se adjuntan el ZIP del proyecto, el notebook principal, la
presentación y el informe Markdown. El ZIP conserva las carpetas necesarias
para ejecutar y consultar las figuras. La copia separada del notebook sirve
para revisar sus salidas; para reentrenar debe abrirse dentro del proyecto.
Al ejecutar se regeneran reportes, figuras y modelo: usar una copia del proyecto
si se desea conservar los resultados de referencia.


- [Informe tecnico](INFORME_TECNICO.md), con el desarrollo del caso,
  evidencias, correspondencia con la rubrica y aportes completos del equipo.
- [Presentacion en PowerPoint](presentation/archivos/Sign_Language_MNIST_base_final.pptx)
  y [notas para organizar la exposicion](presentation/base_presentacion_equipo.md).
- `README.md` con problema, objetivos, KPIs, metodologia, resultados y
  conclusiones.
- Notebook ejecutable en `notebooks/02_sign_language_mnist_mlp_jenaro.ipynb`.
- Carpeta de proyecto ordenada.
- Graficos y evidencias en `images/` y `reports/`.
- Modelo entrenado generado localmente en `models/jenaro_mlp.keras`.
- Presentacion y defensa tecnica de 10 minutos.

## Aporte de Gonzalo Castillo: carga y preprocesamiento

La primera parte implementada corresponde a la preparacion de los datos:

- Lectura de los CSV originales del dataset Sign Language MNIST.
- Separacion de variables de entrada (`pixel1` a `pixel784`) y etiqueta
  (`label`).
- Normalizacion de pixeles desde 0-255 hacia 0-1.
- Transformacion de cada imagen de 28x28 pixeles a un vector de 784 valores
  para el MLP.
- Creacion de una particion de validacion estratificada desde el set de
  entrenamiento.
- Mantencion del set de test oficial para la evaluacion final.

Particiones usadas:

| Conjunto | Filas |
| --- | ---: |
| Entrenamiento | 21.964 |
| Validacion | 5.491 |
| Test oficial | 7.172 |

Para regenerar la evidencia:

```powershell
python scripts\gonzalo_preprocessing_report.py
```

Archivos generados por esta parte:

- `reports/gonzalo_preprocessing_summary.md`
- `images/gonzalo_class_distribution.png`
- `images/gonzalo_sample_grid.png`

## Aporte de Jenaro: modelo y entrenamiento

La version 02 mantiene la preparacion de Gonzalo y desarrolla el modelo MLP
con los contenidos de las actividades 1.4.3 y 1.4.4:

- Tres arquitecturas comparadas: `(32,)`, `(128, 64)` y `(512, 256, 128)`.
- Capas Dense con ReLU y 24 salidas softmax.
- One-hot encoding y `categorical_crossentropy`, como en el ejemplo MNIST.
- Optimizador Adam, learning rate 0.001, batch 128, 20 epocas y semilla 42.
- Comparacion de capacidad sin Dropout, con la misma particion de datos.
- Seleccion por F1 macro de validacion; en empate exacto, menor perdida de
  validacion y menor numero de parametros.

Resultados de validacion de la ejecucion local de referencia:

| Modelo | Capas ocultas | Parametros | Accuracy val | F1 macro val |
| --- | --- | ---: | ---: | ---: |
| A_pequena | 32 | 25.912 | 65.60% | 0.6556 |
| B_mediana | 128, 64 | 110.296 | 97.80% | 0.9790 |
| C_grande | 512, 256, 128 | 569.240 | 100.00% | 1.0000 |

Se selecciono el modelo `C_grande` porque obtuvo el mejor F1 macro de
validacion. Esta seleccion se hizo antes de mirar el test oficial.

Evidencia y explicacion:

- [Informe del modelo y entrenamiento](reports/jenaro_modelo_entrenamiento.md).
- [Comparacion de validacion en CSV](reports/jenaro_comparacion_validacion.csv).
- [Curvas de los tres modelos](images/jenaro_curvas_comparacion.png).
- [Configuracion, mapeo, versiones y huellas](reports/jenaro_entrenamiento.json).
- [Apoyo para la defensa de Jenaro](presentation/jenaro_defensa.md).

## Aporte de Jason: evaluacion y analisis de errores

Despues de seleccionar el modelo por validacion, se evaluo el modelo `C_grande`
en el set de test oficial.

Resultados de test:

| Metrica | Valor |
| --- | ---: |
| Accuracy test | 79.71% |
| Precision macro test | 0.7783 |
| Recall macro test | 0.7859 |
| F1 macro test | 0.7760 |
| Imagenes evaluadas | 7.172 |

La diferencia entre el 100.00% de validacion y el 79.71% de test muestra que
la validacion interna fue optimista. El modelo aprende patrones utiles, pero
no mantiene el mismo rendimiento sobre el test oficial.

Letras con mejor desempeno:

- B: F1-score 0.97.
- A y E: F1-score 0.94.
- P: F1-score 0.91.
- C: F1-score 0.90.

Letras con peor desempeno:

- S: F1-score 0.52.
- R: F1-score 0.58.
- T: F1-score 0.59.
- N: F1-score 0.61.
- U: F1-score 0.63.

Los errores se concentran en letras visualmente parecidas. En los ejemplos
mal clasificados aparecen confusiones como B con U, D con X, N con A, S con I,
T con X, N con S, U con D, U con Y y F con C.

Limitacion principal del MLP: al usar una entrada plana de 784 valores, el
modelo conserva los pixeles, pero no aprovecha explicitamente la relacion
espacial entre pixeles vecinos como lo haria una CNN. Por eso puede ser mas
sensible a cambios de posicion, fondo o pequenas diferencias entre manos.

Evidencia:

- [Metricas de test](reports/jenaro_integracion_test.json).
- [Reporte de clasificacion por clase](reports/jenaro_integracion_classification_report.txt).
- [Matriz de confusion](images/jenaro_integracion_confusion_matrix.png).
- [Ejemplos correctos](images/jenaro_integracion_correct_examples.png).
- [Ejemplos incorrectos](images/jenaro_integracion_incorrect_examples.png).

## Division de trabajo

### Gonzalo

- Preparar la carga del dataset.
- Separar datos de entrenamiento, validacion y prueba.
- Normalizar pixeles.
- Convertir las imagenes al formato que necesita el MLP.
- Documentar la descripcion del dataset, objetivos, KPIs, CRISP-DM y
  preprocesamiento.

### Jenaro

- Implementar la arquitectura MLP.
- Definir capas ocultas, neuronas, activaciones, funcion de salida, perdida y
  optimizador.
- Entrenar una version base y comparar arquitecturas.
- Documentar decisiones tecnicas del modelo y entrenamiento.

### Jason

- Interpretar distribucion de clases y ejemplos visuales.
- Calcular metricas: accuracy, precision, recall y F1-score.
- Construir e interpretar la matriz de confusion.
- Seleccionar ejemplos bien y mal clasificados.
- Redactar analisis de errores, limitaciones del MLP y conclusiones.

### Todos

- Revisar que el notebook se ejecute completo.
- Ensayar preguntas cruzadas para la defensa.
- Asegurarse de que cada integrante pueda explicar el flujo completo.

## Estructura del repositorio

```text
sign-language-mnist-mlp/
  data/
    raw/              # CSV originales extraidos del ZIP
    processed/        # Datos preparados si se guardan
  images/             # Graficos, matriz de confusion y ejemplos
  models/             # Modelo entrenado localmente
  notebooks/          # Notebook principal
  presentation/       # Guion o apoyo para exponer
  reports/            # Resultados y tablas exportadas
  scripts/            # Scripts auxiliares
  src/sign_mlp/       # Codigo reutilizable del proyecto
```

## Instalacion y ejecucion

Entorno de verificación: Python 3.13.6 en Windows, con las dependencias fijadas.
El notebook utiliza `src/` y los CSV del proyecto; descomprimir primero el ZIP completo.

Para ejecutar localmente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-jenaro.txt
```

Los CSV ya vienen incluidos en `data/raw/` al clonar o descargar el repositorio
completo. No es necesario extraer un ZIP ni modificar rutas personales.
Ver [archivos e integridad del dataset](data/README.md).

Despues abrir y ejecutar el notebook principal:

```powershell
jupyter notebook notebooks\02_sign_language_mnist_mlp_jenaro.ipynb
```

Para ejecutar y guardar las salidas desde terminal:

```powershell
jupyter nbconvert --to notebook --execute notebooks\02_sign_language_mnist_mlp_jenaro.ipynb --output 02_sign_language_mnist_mlp_ejecutado.ipynb --ExecutePreprocessor.timeout=1800
```

El repositorio incluye los dos CSV originales. El modelo binario se genera al
ejecutar el notebook y permanece fuera de Git. Se necesitan las dependencias
de Python indicadas arriba, pero no una descarga adicional del dataset.

## Metodologia CRISP-DM

1. Comprension del negocio: reconocer letras de lenguaje de senas como apoyo
   educativo o de accesibilidad.
2. Comprension de los datos: analizar cantidad de clases, ejemplos por clase y
   formato de imagen.
3. Preparacion de datos: normalizar pixeles, separar entrenamiento, validacion
   y prueba, y adaptar las imagenes a vectores.
4. Modelamiento: implementar y comparar MLP con capas densas.
5. Evaluacion: revisar metricas, curvas de entrenamiento, matriz de confusion y
   ejemplos correctos e incorrectos.
6. Documentacion: dejar repositorio reproducible y explicar limitaciones.

## Preguntas que deben saber responder

- Por que se eligio este dataset.
- Que problema de clasificacion resuelve el modelo.
- Por que el problema es multiclase y no binario.
- Que es un MLP y como procesa una imagen.
- Por que las imagenes se transforman de 28x28 a un vector de 784 valores.
- Que significan accuracy, precision, recall, F1-score y matriz de confusion.
- Por que se usa softmax con 24 salidas.
- Por que se usa one-hot encoding con `categorical_crossentropy`.
- Que diferencia hay entre train, validacion y test.
- Que significa que el modelo tenga 100% en validacion pero 79.71% en test.
- Por que un MLP no es la mejor arquitectura para imagenes.
- Que mejoras se podrian implementar despues, por ejemplo CNN, data augmentation
  o ajuste de hiperparametros.

## Checklist de rubrica

- [x] Problema definido y dataset justificado.
- [x] Variante propia del trabajo documentada.
- [x] Carga y preprocesamiento explicado.
- [x] MLP implementado y justificado.
- [x] Entrenamiento y validacion registrados.
- [x] Accuracy, precision, recall, F1-score y matriz de confusion calculados.
- [x] Ejemplos correctos e incorrectos analizados.
- [x] Limitaciones del MLP explicadas.
- [x] README actualizado con resultados actuales.
- [ ] Presentacion ensayada por los tres integrantes.
