# Revisión del bloque de Jenaro para la EP2

El bloque comprende arquitectura CNN, entrenamiento e interpretabilidad,
secciones 4 y 5 del informe y diapositivas 4 a 6. La revisión parte del commit
`0411154d7711ef113853f5377d0c3b5922b0ebec` de la rama `ep2-cnn`.
La arquitectura y los resultados de referencia ya estaban en la base compartida.

## Cambios

- Se mantiene la arquitectura de 243.352 parámetros y su protocolo de
  entrenamiento. Se explican dimensiones, filtros compartidos, ReLU, pérdida
  sparse, Adam, Dropout y callbacks en las secciones 4 y 5 del informe.
- `export_run` unifica la exportación del notebook y del script. Guarda resumen,
  historial, métricas, todas las figuras y modelo de la misma corrida. Los dos
  JSON de métricas tienen ahora el mismo contenido e identifican versiones.
- `--output-dir` y la variable `EP2_OUTPUT_DIR` del notebook permiten guardar
  una ejecución aparte de las evidencias históricas.
- Los Feature Maps incluyen la imagen de entrada, resolución de cada capa y
  los primeros ocho canales, con escala individual.
- Grad-CAM usa el puntaje anterior a softmax y conserva las predicciones del
  modelo. Los mapas se amplían y superponen a la entrada siguiendo el
  [método ilustrado por Keras](https://keras.io/examples/vision/grad_cam/).
- Se agregan mapas para las 24 clases. Cada clase tiene una muestra real;
  se selecciona el primer acierto o, si no existe, su primer error. El mapa
  explica esa clase real. El panel de seis ejemplos explica la clase predicha
  y conserva aciertos y errores.
- Se prepara una guía individual EP2 con guion de 3:10 y preguntas de defensa.
  La guía de EP1 se conserva.

## Ejecución local

Se ejecutaron las **16 celdas de código** del notebook, sin excepciones, con
semilla 42, las particiones originales y el test oficial completo.

| Dato | Resultado local |
| --- | ---: |
| Imágenes train / validación / test | 21.964 / 5.491 / 7.172 |
| Épocas | 20 |
| Parámetros de la CNN | 243.352 |
| Tiempo de entrenamiento | 127,95 s |
| Accuracy de validación | 100,00% |
| Accuracy de test | 96,81% |
| Precisión macro de test | 0,9630 |
| Recall macro de test | 0,9650 |
| F1 macro de test | 0,9630 |

El entorno utiliza Python 3.12.10, TensorFlow 2.20.0, Keras 3.15.1 y NumPy
2.3.5. Las versiones completas están en
[`requirements-lock.txt`](../output/jenaro_ep2/requirements-lock.txt).
TensorFlow necesitó las bibliotecas de ejecución C++ de Windows; se habilitaron
localmente dentro de `.venv` usando el runtime ya incluido en este equipo.

Los callbacks redujeron el learning rate al cerrar las épocas 14, 17 y 20.
La última tasa, 0,000125, no se utilizó en otra época. Se restauraron los pesos
de la época 19; no hubo parada anticipada.

La referencia del equipo mantiene 96,60% de test, F1 macro 0,9595 y 41,41 s.
La corrida anterior del notebook registra 45,49 s. Esta nueva ejecución usa
otro entorno; no permite atribuir diferencias a un único factor ni comparar
velocidad con el MLP histórico de 31,50 s. La arquitectura no se cambió ni
se ajustó con los resultados del test.

## Evidencias

- [Notebook ejecutado](../output/jenaro_ep2/03_sign_language_mnist_cnn_ep2_ejecutado.ipynb).
- [Métricas de la corrida local](../output/jenaro_ep2/reports/cnn_metrics.json).
- [Resumen del modelo](../output/jenaro_ep2/reports/cnn_model_summary.txt) e
  [historial](../output/jenaro_ep2/reports/cnn_history.csv).
- [Curvas](../output/jenaro_ep2/images/cnn_training_curves.png).
- [Feature Maps](../output/jenaro_ep2/images/cnn_feature_maps.png).
- [Grad-CAM de aciertos y errores](../output/jenaro_ep2/images/cnn_gradcam_examples.png).
- [Grad-CAM de las 24 clases](../output/jenaro_ep2/images/cnn_gradcam_by_class.png).
- [Guía de defensa EP2](../presentation/jenaro_defensa_ep2.md).
- [Presentación con el bloque de Jenaro revisado](../presentation/archivos/Sign_Language_MNIST_CNN_EP2_Jenaro.pptx).

La copia de la presentación revisa únicamente las diapositivas 4 a 6 y sus
notas, conserva las diez diapositivas y las cifras de referencia del equipo.
En la diapositiva 6 se usan dos ejemplos superpuestos de la nueva corrida
como ilustración, identificados en notas: acierto G/G y error N/A.

El panel de 24 clases se volvió a renderizar desde los pesos guardados para
ampliar el espacio entre filas; su imagen también se actualizó en la salida
del notebook, sin repetir ni modificar el entrenamiento. Cada mapa se
normaliza por imagen y tiene resolución original 7x7. Puede destacar mano,
bordes o fondo: no es una medida de confianza ni una prueba causal.

El modelo entrenado queda en
`output/jenaro_ep2/models/cnn_sign_language.keras`, excluido de Git porque se
reconstruye con los datos y el código. Las evidencias originales de `reports/`
e `images/` conservan sus valores.

## Comprobaciones y reproducción

Las cuatro pruebas de `tests/test_cnn_interpretability.py` pasan: mapa espacial
con softmax saturado sin modificar predicciones, elección de clase con
activación nula finita, selección de todas las clases incluso con errores y
conservación del mapa después de guardar y cargar el modelo. Se recorre la
CNN secuencialmente para que los gradientes sigan conectados también al cargar
los pesos guardados, sin depender de nodos simbólicos duplicados de Keras.
La ejecución real verifica además cobertura de las 24 letras, exportación
completa y JSON coincidentes. `git diff --check` no detecta errores.
La celda de inicio también se comprobó desde la carpeta anidada del notebook
ejecutado. El [registro de validación](../output/jenaro_ep2/validation.json)
conserva estos resultados y las huellas SHA-256 de los CSV utilizados.

Desde la raíz del proyecto y con el entorno preparado:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\train_cnn_ep2.py --output-dir output\jenaro_ep2
```

Para ejecutar el notebook en otra carpeta de salida, definir
`EP2_OUTPUT_DIR` antes de iniciar Jupyter. Sin esa variable, exporta en la raíz
del proyecto. Las cifras de README, informe y PPT deben identificar la corrida
elegida antes de la entrega del equipo.

El bloque se entrega sobre la rama `ep2-cnn`. Las instrucciones para continuar
e integrar los resultados están en el README.
