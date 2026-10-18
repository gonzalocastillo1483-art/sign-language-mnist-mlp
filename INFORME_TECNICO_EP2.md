# Informe tecnico EP2: clasificacion de lenguaje de senas con CNN

## 1. Integrantes y objetivo

**Integrantes:** Gonzalo Castillo, Jenaro Marin y Jason Aguilera.

El objetivo de la EP2 es implementar una red neuronal convolucional (CNN) para
resolver el mismo problema multiclase de la EP1 y comparar objetivamente su
desempeno con el MLP desarrollado anteriormente.

El caso corresponde al reconocimiento de letras estaticas del alfabeto ASL a
partir de imagenes de manos. Una aplicacion posible es apoyar herramientas
educativas o de accesibilidad que necesiten identificar una sena individual.

## 2. Continuidad y variante seleccionada

Se reutiliza el dataset **Sign Language MNIST** y se conserva la variante de
clasificacion multiclase de 24 letras. Esta decision permite que la diferencia
principal entre los experimentos sea el tipo de red: MLP en la EP1 y CNN en la
EP2.

La comparacion es pertinente porque el MLP recibe un vector plano y no modela
explicitamente la cercania entre pixeles. La CNN conserva la estructura 2D de
la imagen y aprende filtros locales que detectan bordes, contornos y patrones
de mayor nivel.

## 3. Datos y preparacion para la CNN

Los archivos `sign_mnist_train.csv` y `sign_mnist_test.csv` contienen una
columna `label` y 784 columnas de intensidad. Cada fila representa una imagen
de 28 x 28 pixeles en escala de grises.

Esta etapa no repite la limpieza ni el EDA completo realizado en la EP1. Se
reutiliza el conjunto ya validado y se documentan solamente las verificaciones
y transformaciones necesarias para cambiar desde la entrada plana del MLP al
formato tensorial requerido por la CNN.

El conjunto incluye 24 clases: A-I y K-Y. J y Z se excluyen porque sus senas
requieren movimiento, mientras que el dataset contiene imagenes estaticas.

Se aplicaron los siguientes pasos:

1. Lectura de los CSV y separacion de pixeles y etiquetas.
2. Division estratificada del archivo de entrenamiento en 21.964 imagenes de
   entrenamiento y 5.491 de validacion.
3. Conservacion de las 7.172 imagenes del test oficial para la evaluacion final.
4. Normalizacion de cada pixel dividiendo por 255, desde el rango 0-255 al
   rango 0-1.
5. Reconstruccion de cada fila como tensor `(28, 28, 1)`, donde el ultimo valor
   representa el canal de escala de grises.

La estratificacion mantiene aproximadamente la proporcion de las 24 clases en
entrenamiento y validacion. No hay etiquetas fuera del mapeo esperado ni
pixeles fuera del rango valido. No se utilizo data augmentation en esta version
para mantener una comparacion simple y reproducible; puede incorporarse en un
experimento posterior.

## 4. Arquitectura y decisiones de diseno

La arquitectura implementada es:

| Etapa | Configuracion | Justificacion |
| --- | --- | --- |
| Entrada | 28 x 28 x 1 | Conserva la estructura espacial de la imagen gris. |
| Bloque 1 | Conv2D 32, 3x3, ReLU + MaxPool 2x2 | Aprende patrones locales basicos y reduce dimensiones. |
| Bloque 2 | Conv2D 64, 3x3, ReLU + MaxPool 2x2 | Combina bordes en formas mas complejas. |
| Regularizacion | Dropout 0,25 | Reduce coadaptacion entre activaciones. |
| Bloque 3 | Conv2D 128, 3x3, ReLU + MaxPool 2x2 | Aprende representaciones de mayor nivel. |
| Clasificador | Flatten + Dense 128 ReLU + Dropout 0,40 | Integra las caracteristicas antes de clasificar. |
| Salida | Dense 24, softmax | Produce una probabilidad para cada clase. |

Se uso `sparse_categorical_crossentropy` porque las etiquetas se conservan como
indices enteros y el problema tiene una sola clase correcta por imagen. No se
uso perdida binaria porque las clases son mutuamente excluyentes y existen 24
posibles resultados.

El optimizador fue Adam con learning rate inicial de 0,001, batch de 128 y un
maximo de 20 epocas. `EarlyStopping` restaura los mejores pesos si la perdida de
validacion deja de mejorar, y `ReduceLROnPlateau` reduce el paso de aprendizaje
cuando el avance se estanca. La semilla 42 facilita la reproducibilidad.

## 5. Entrenamiento e interpretabilidad

El modelo CNN tiene 243.352 parametros entrenables. En la ejecucion de
referencia completo 20 epocas en 41,41 segundos y alcanzo 100% de accuracy de
validacion.

Las curvas muestran que la perdida disminuye y el accuracy aumenta sin una
separacion sostenida entre entrenamiento y validacion. Los mapas de
caracteristicas muestran que las primeras capas responden a bordes y siluetas,
mientras que las capas profundas conservan regiones mas abstractas de la mano.
Los ejemplos Grad-CAM permiten observar que la decision se concentra
principalmente sobre la forma de la mano y no sobre el fondo completo.

Evidencia:

- `images/cnn_training_curves.png`
- `images/cnn_feature_maps.png`
- `images/cnn_gradcam_examples.png`
- `reports/cnn_history.csv`
- `reports/cnn_model_summary.txt`

## 6. Evaluacion en test

| Metrica | Resultado CNN |
| --- | ---: |
| Accuracy | 96,60% |
| Precision macro | 0,9604 |
| Recall macro | 0,9616 |
| F1 macro | 0,9595 |
| Imagenes evaluadas | 7.172 |

Se reporta F1 macro junto con accuracy porque el primero entrega el mismo peso
a cada clase. Esto evita que el buen rendimiento de una clase con mas ejemplos
oculte el desempeno de otra clase.

Las clases W, P, O, F y D obtuvieron F1 de 1,0. Las clases con menor F1 fueron
T (0,7745), X (0,8586), R (0,8723), Q (0,9086) y L (0,9187).

Las principales confusiones fueron T con X, G con Q, X con T, S con E y R con
L. Estos pares comparten posiciones de dedos o siluetas similares, por lo que
una pequena diferencia visual puede cambiar la prediccion.

Evidencia:

- `reports/cnn_metrics.json`
- `reports/cnn_classification_report.txt`
- `images/cnn_confusion_matrix.png`
- `images/cnn_correct_examples.png`
- `images/cnn_incorrect_examples.png`

## 7. Comparacion objetiva MLP versus CNN

| Metrica | MLP EP1 | CNN EP2 | Interpretacion |
| --- | ---: | ---: | --- |
| Parametros | 569.240 | 243.352 | La CNN usa 57,25% menos. |
| Tiempo | 31,50 s | 41,41 s | La CNN tarda 9,91 s mas en esta maquina. |
| Accuracy validacion | 100,00% | 100,00% | Ambos ajustan muy bien el split interno. |
| Accuracy test | 79,71% | 96,60% | La CNN mejora 16,89 puntos porcentuales. |
| F1 macro test | 0,7760 | 0,9595 | La CNN mejora de forma consistente entre clases. |
| Brecha validacion-test | 20,29 pp | 3,40 pp | La CNN generaliza mucho mejor. |

La mejora no se atribuye a un cambio de dataset: ambos modelos usan las mismas
imagenes, normalizacion y particiones. La diferencia se explica principalmente
por el sesgo inductivo de las convoluciones, que aprovechan patrones locales y
comparten pesos a lo largo de la imagen.

## 8. Limitaciones y mejoras posibles

- El test oficial puede representar condiciones distintas al split interno de
  validacion, aunque la CNN reduce considerablemente la brecha.
- El modelo reconoce letras estaticas y no puede representar J, Z o palabras en
  movimiento.
- Algunas senas tienen diferencias muy pequenas entre dedos y orientacion.
- No se midio robustez con camaras, fondos, iluminaciones o personas externas.
- Grad-CAM entrega una explicacion aproximada y no demuestra causalidad.

Como trabajo futuro se propone probar data augmentation, Batch Normalization,
busqueda sistematica de hiperparametros, validacion cruzada y un conjunto de
imagenes externas. Para senas dinamicas seria necesario modelar secuencias de
video.

## 9. Conclusiones

La CNN fue adecuada para este problema porque conserva la estructura espacial
de las imagenes y aprende caracteristicas locales. Frente al MLP, consiguio un
accuracy de test 16,89 puntos mayor y un F1 macro 0,1835 superior con menos de
la mitad de los parametros. Su costo fue un tiempo de entrenamiento algo mayor.

El resultado principal no es solamente el 96,60% de accuracy, sino la mejora
en generalizacion: la brecha entre validacion y test se redujo de 20,29 a 3,40
puntos porcentuales. Los errores restantes se concentran en senas visualmente
parecidas, lo que orienta las proximas mejoras del proyecto.

## 10. Reproducibilidad y aportes

El flujo completo esta en `notebooks/03_sign_language_mnist_cnn_ep2.ipynb` y
tambien puede ejecutarse con `python scripts/train_cnn_ep2.py`. Las dependencias
estan registradas en `requirements-ep2.txt`.

- **Gonzalo Castillo:** continuidad, variante y adaptacion de vector MLP a tensor CNN.
- **Jenaro Marin:** arquitectura, entrenamiento e interpretabilidad.
- **Jason Aguilera:** evaluacion, comparacion, errores, limitaciones y cierre.

Los tres integrantes revisan la ejecucion final, la consistencia de los
resultados y la defensa cruzada del proyecto completo.
