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
- [ ] MLP implementado y justificado.
- [ ] Entrenamiento y validacion registrados.
- [ ] Accuracy, precision, recall, F1-score y matriz de confusion calculados.
- [ ] Ejemplos correctos e incorrectos analizados.
- [ ] Limitaciones del MLP explicadas.
- [ ] README y notebook limpios.
- [ ] Presentacion ensayada por los tres integrantes.
