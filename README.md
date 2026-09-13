# Clasificacion de lenguaje de senas con MLP

Proyecto para la Evaluacion Parcial 1 de TLY1102. El objetivo es construir, entrenar y defender un modelo de red neuronal tipo MLP para clasificar imagenes del dataset Sign Language MNIST.

## Decision del equipo

Dataset elegido: `Sign Language MNIST.zip`.

Variante inicial: clasificar las letras disponibles del alfabeto ASL usando imagenes de 28x28 pixeles en escala de grises. Si el tiempo no alcanza, se puede reducir a un subconjunto de clases, por ejemplo A-F o A-L, dejando documentada la decision.

## Problema de negocio

Una aplicacion educativa o de accesibilidad necesita reconocer senas de letras del alfabeto a partir de imagenes. El modelo propuesto busca clasificar cada imagen en la letra correspondiente, como primera version de un sistema de apoyo para aprendizaje de lenguaje de senas.

## Objetivos

- Implementar un flujo completo de ciencia de datos para clasificacion de imagenes.
- Entrenar una red neuronal MLP como modelo base.
- Evaluar el desempeno con metricas de clasificacion.
- Analizar errores y limitaciones del modelo.
- Documentar el proceso de forma reproducible.

## KPIs

- Accuracy de validacion.
- F1-score macro para medir desempeno balanceado entre clases.
- Precision y recall por clase.
- Diferencia entre accuracy de entrenamiento y validacion para observar posible sobreajuste.
- Cantidad de clases con mayor confusion en la matriz de confusion.

## Entregables

- `README.md` con problema, objetivos, KPIs, metodologia, resultados y conclusiones.
- Notebook ejecutable en `notebooks/01_sign_language_mnist_mlp.ipynb`.
- Carpeta de proyecto ordenada.
- Graficos y evidencias en `images/` o `reports/`.
- Modelo entrenado guardado en `models/` si corresponde.
- Presentacion y defensa tecnica de 10 minutos.

## Avance actual: Gonzalo Castillo

La primera parte implementada corresponde a la carga y preparacion de datos:

- Lectura de los CSV originales del dataset Sign Language MNIST.
- Separacion de variables de entrada (`pixel1` a `pixel784`) y etiqueta (`label`).
- Normalizacion de pixeles desde 0-255 hacia 0-1.
- Transformacion de cada imagen de 28x28 pixeles a un vector de 784 valores para el MLP.
- Creacion de una particion de validacion estratificada desde el set de entrenamiento.
- Mantencion del set de test oficial para la evaluacion final.

Para regenerar la evidencia:

```powershell
python scripts\gonzalo_preprocessing_report.py
```

Archivos generados por esta parte:

- `reports/gonzalo_preprocessing_summary.md`
- `images/gonzalo_class_distribution.png`
- `images/gonzalo_sample_grid.png`

## Aporte de Jenaro: modelo y entrenamiento

El propósito académico es comprender el caso y fundamentar las decisiones.
Las métricas sirven para discutir qué aprendió el modelo, dónde falla y
para qué podría resultar útil. Se conserva el experimento realizado y sus
resultados, aunque muestren limitaciones. El trabajo se realiza directamente
en `main`, según la indicación de Jenaro, sin crear ramas adicionales.

El dataset fue confirmado por Jenaro como aprobado por el profesor. La
versión con su aporte es
[`notebooks/02_sign_language_mnist_mlp_jenaro.ipynb`](notebooks/02_sign_language_mnist_mlp_jenaro.ipynb).
El notebook 01 se conserva como versión inicial del equipo.

La versión 02 mantiene la preparación de Gonzalo y desarrolla las secciones
5 y 6 con los contenidos de las actividades 1.4.3 y 1.4.4:

- Tres arquitecturas: `(32,)`, `(128, 64)` y `(512, 256, 128)`.
- Capas Dense con ReLU y 24 salidas softmax.
- One-hot encoding y `categorical_crossentropy`, como en el ejemplo MNIST.
- Adam, learning rate 0,001, batch 128, 20 épocas y semilla 42.
- Comparación de capacidad sin Dropout, con la misma partición de datos.
- Selección por F1 macro de validación; en empate exacto, menor pérdida de
  validación y menor número de parámetros. Se comparan los pesos de la última época.

El test oficial se evalúa después de la selección, utilizando el código de
evaluación que ya tenía el proyecto. Las métricas y figuras automáticas
sirven como insumo para Jason; su interpretación de errores y las conclusiones
globales siguen siendo su aporte. La variante propia del equipo debe
documentarse entre los integrantes; se conserva el split existente de Gonzalo.

Evidencia y explicación:

- [Informe del modelo y entrenamiento](reports/jenaro_modelo_entrenamiento.md).
- [Comparación de validación en CSV](reports/jenaro_comparacion_validacion.csv).
- [Curvas de los tres modelos](images/jenaro_curvas_comparacion.png).
- [Configuración, mapeo, versiones y huellas](reports/jenaro_entrenamiento.json).
- [Apoyo para la defensa de Jenaro](presentation/jenaro_defensa.md).

Resultados de la ejecución local de referencia (20 épocas por modelo):

| Capas ocultas | Parámetros | Accuracy de validación | F1 macro de validación |
| --- | ---: | ---: | ---: |
| 32 | 25.912 | 65,60% | 0,6556 |
| 128, 64 | 110.296 | 97,80% | 0,9790 |
| 512, 256, 128 | 569.240 | 100,00% | 1,0000 |

Se seleccionó el modelo de 512, 256 y 128 neuronas. Su evaluación posterior
en el test oficial obtuvo **79,17% de accuracy y 0,7703 de F1 macro**.
La caída de 20,83 puntos respecto de validación muestra que el 100% interno
no se mantiene en test. El informe explica este límite sin atribuir causas
no comprobadas ni volver a seleccionar el modelo usando test.

El modelo seleccionado se guarda localmente en `models/jenaro_mlp.keras`.
Los CSV y el modelo binario siguen excluidos de Git; al compartir solamente
el repositorio deben proporcionarse los datos y volver a ejecutar el notebook.
El modelo espera 784 píxeles divididos por 255. Su pérdida requiere etiquetas
one-hot al llamar a `evaluate`. `predict` devuelve probabilidades y no recibe etiquetas.

### Ejecutar el aporte desde cero

Desde la raíz del proyecto, con Python y el ZIP disponible:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-jenaro.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (Sign MLP)"
python scripts/prepare_data.py --zip "RUTA/Sign Language MNIST.zip"
jupyter notebook notebooks/02_sign_language_mnist_mlp_jenaro.ipynb
```

En Jupyter: reiniciar el kernel y ejecutar todas las celdas en orden. Para
ejecutar y guardar las salidas desde terminal con el entorno activado:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/02_sign_language_mnist_mlp_jenaro.ipynb --ExecutePreprocessor.timeout=1800
```

Se necesita el repositorio completo porque el notebook importa módulos de
`src/sign_mlp/`. En Colab, colocar también `src/` y los CSV en sus carpetas,
trabajar desde la raíz y usar el notebook 02. Las versiones de Colab pueden
diferir de las registradas para la ejecución local.

Los originales de clases se conservan fuera del repositorio. La metodología
proviene de esos materiales; no se agregan CNN, data augmentation ni búsquedas
automáticas. Las métricas se obtienen mediante ejecución real. La defensa
debe fundamentar las decisiones a partir de esos resultados.

## Division de trabajo

### Gonzalo

- Preparar la carga del dataset.
- Separar datos de entrenamiento, validacion y prueba.
- Normalizar pixeles.
- Convertir las imagenes al formato que necesita el MLP.
- Documentar en README la descripcion del dataset, objetivos, KPIs, CRISP-DM y preprocesamiento.

### Jenaro

- Implementar la arquitectura MLP.
- Definir capas ocultas, neuronas, activaciones, funcion de salida, perdida y optimizador.
- Entrenar una version base.
- Comparar 2 o 3 configuraciones simples de hiperparametros.
- Documentar decisiones tecnicas del modelo y entrenamiento.

### Jason

- Realizar EDA: distribucion de clases y ejemplos visuales.
- Calcular metricas: accuracy, precision, recall y F1-score.
- Construir matriz de confusion.
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
  models/             # Modelo entrenado
  notebooks/          # Notebook principal
  presentation/       # Guion o apoyo para exponer
  reports/            # Resultados y tablas exportadas
  scripts/            # Scripts auxiliares
  src/sign_mlp/       # Codigo reutilizable del proyecto
```

## Instalacion

Se recomienda trabajar en Google Colab si TensorFlow da problemas en Windows.

Para ejecutar localmente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Luego preparar el dataset:

```powershell
python scripts\prepare_data.py --zip "C:\Users\gonza\Downloads\Sign Language MNIST.zip"
```

Despues abrir y ejecutar:

```powershell
jupyter notebook notebooks\01_sign_language_mnist_mlp.ipynb
```

## Metodologia CRISP-DM

1. Comprension del negocio: reconocer letras de lenguaje de senas como apoyo educativo o de accesibilidad.
2. Comprension de los datos: analizar cantidad de clases, ejemplos por clase y formato de imagen.
3. Preparacion de datos: normalizar pixeles, separar entrenamiento/validacion/prueba y adaptar las imagenes a vectores.
4. Modelamiento: implementar un MLP con capas densas.
5. Evaluacion: revisar metricas, curvas de entrenamiento y matriz de confusion.
6. Despliegue/documentacion: dejar repositorio reproducible y explicar limitaciones.

## Preguntas que deben saber responder

- Por que se eligio este dataset.
- Que problema de clasificacion resuelve el modelo.
- Que es un MLP y como procesa una imagen.
- Por que las imagenes se transforman de 28x28 a un vector de 784 valores.
- Que significan accuracy, precision, recall, F1-score y matriz de confusion.
- Que indica el sobreajuste en las curvas de entrenamiento.
- Por que un MLP no es la mejor arquitectura para imagenes.
- Que mejoras se podrian implementar despues, por ejemplo CNN, data augmentation o ajuste de hiperparametros.

## Checklist de rubrica

- [ ] Problema definido y dataset justificado.
- [ ] Variante propia del trabajo documentada.
- [ ] Carga y preprocesamiento explicado.
- [x] MLP implementado y justificado (aporte de Jenaro, notebook 02).
- [x] Entrenamiento y validacion registrados (aporte de Jenaro, notebook 02).
- [ ] Accuracy, precision, recall, F1-score y matriz de confusion calculados.
- [ ] Ejemplos correctos e incorrectos analizados.
- [ ] Limitaciones del MLP explicadas.
- [ ] README y notebook limpios.
- [ ] Presentacion ensayada por los tres integrantes.
