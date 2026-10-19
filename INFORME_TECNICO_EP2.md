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

Las convoluciones usan `padding="same"` y paso 1: mantienen el alto y ancho
antes del pooling. El flujo espacial es 28x28 -> 14x14 -> 7x7 -> 3x3; el
ultimo pooling usa ventanas completas, por lo que 7 se reduce a 3. `Flatten`
convierte 3x3x128 activaciones en 1.152 valores para la capa densa.

Un filtro 3x3 aprende relaciones locales con pocos pesos. Los filtros se
comparten en todas las posiciones; por ejemplo, la primera convolucion usa
(3x3x1 + 1)x32 = 320 parametros, incluidos los sesgos. El crecimiento
32 -> 64 -> 128 aumenta la cantidad de patrones disponibles mientras el
pooling reduce el costo espacial. Es una decision de diseno de esta version,
no el resultado de una busqueda que demuestre una arquitectura optima.

ReLU calcula `max(0, x)` e introduce no linealidad entre las capas. Softmax
normaliza las 24 salidas para elegir una clase mutuamente excluyente. El MLP
conserva los valores de los pixeles al aplanar, pero sus capas densas no
incorporan esta estructura de filtros locales compartidos.

Se uso `sparse_categorical_crossentropy` porque las etiquetas se conservan como
indices enteros contiguos de 0 a 23, y el problema tiene una sola clase
correcta por imagen. El mapeo a las etiquetas originales se conserva para
mostrar las letras; no se utiliza one-hot en la CNN. No se
uso perdida binaria porque las clases son mutuamente excluyentes y existen 24
posibles resultados.

El optimizador fue Adam con learning rate inicial de 0,001, batch de 128 y un
maximo de 20 epocas. Adam mantiene estimaciones del promedio y de la magnitud
de los gradientes para adaptar las actualizaciones por parametro. El batch
produce 172 actualizaciones por epoca, incluida la ultima incompleta.

`EarlyStopping` observa `val_loss`, con paciencia de 4 epocas y restauracion
de los mejores pesos. `ReduceLROnPlateau` observa la misma perdida y reduce
el learning rate a la mitad tras 2 epocas sin mejora suficiente, con minimo
de 0,00001. Ambos callbacks usan sus umbrales predeterminados de mejora, que
no son identicos. Dropout usa tasas de 0,25 y 0,40 solo durante el entrenamiento;
en inferencia y validacion se desactiva. La semilla 42 y las operaciones
deterministas facilitan repetir una corrida dentro del mismo entorno. Cambios
de versiones o hardware pueden modificar numeros y tiempos.

## 5. Entrenamiento e interpretabilidad

El modelo CNN tiene 243.352 parametros entrenables. En la ejecucion de
referencia completo 20 epocas en 41,41 segundos y alcanzo 100% de accuracy de
validacion. El resumen posterior al entrenamiento tambien incluye estados
del optimizador Adam; esos estados no son parametros aprendidos adicionales
de la arquitectura. El notebook previo registra otra ejecucion de 45,49 s;
esa duracion no sustituye los 41,41 s de la referencia usada en la comparacion.

En el historial de referencia, la perdida de entrenamiento baja de 2,3968 a
0,0157 y la accuracy de 26,83% a 99,43%. La validacion alcanza 100% desde la
epoca 7. El learning rate pasa de 0,001 a 0,0005 y luego a 0,00025;
al cerrar la epoca 20 el callback lo reduce otra vez a 0,000125, sin utilizar
esa ultima tasa en una nueva epoca. EarlyStopping no detiene antes del limite
de 20 epocas. La menor `val_loss`
se alcanza en la epoca 19 (indice 18 del CSV), cuyos pesos se restauran.

La validacion supera al entrenamiento en buena parte de las curvas. Esto es
compatible con Dropout activo en train e inactivo en validacion, y con que la
metrica de entrenamiento acumula lotes mientras los pesos cambian. Las curvas
no bastan para descartar sobreajuste ni demostrar generalizacion fuera del
dataset; Jason analiza el test independiente.

Los Feature Maps muestran los primeros ocho canales de `conv_1`, `conv_2` y
`conv_3`, junto a la imagen original. Sus resoluciones son 28x28, 14x14 y 7x7
antes del pooling correspondiente. Las respuestas pueden asociarse a
contrastes, contornos y combinaciones mas selectivas; no se asigna un
significado fijo a cada filtro a partir de una sola imagen.

Grad-CAM pondera estos mapas con el gradiente del puntaje de una clase antes
de softmax, sin cambiar el clasificador. Los mapas 7x7 se amplian y se
superponen a la imagen para localizar las activaciones. Se explican tanto
aciertos como errores para la clase predicha. Ademas se exporta un panel de
las 24 clases, con una muestra por clase real y Grad-CAM dirigido a esa clase.
La seleccion toma el primer acierto disponible, o el primer error si no hay
aciertos. Cada mapa se normaliza por separado: sus colores no permiten
comparar confianza entre imagenes y no prueban causalidad. Las activaciones
pueden incluir partes de la mano y tambien bordes o fondo.

Evidencia:

- `images/cnn_training_curves.png`
- `images/cnn_feature_maps.png`
- `images/cnn_gradcam_examples.png`
- `reports/cnn_history.csv`
- `reports/cnn_model_summary.txt`

La revision de Jenaro conserva las cifras de referencia del equipo. La nueva
ejecucion y sus mapas completos quedan identificados por separado en
`output/jenaro_ep2/`, junto al notebook ejecutado y las versiones utilizadas.
Sus figuras son `output/jenaro_ep2/images/cnn_training_curves.png`,
`output/jenaro_ep2/images/cnn_feature_maps.png`,
`output/jenaro_ep2/images/cnn_gradcam_examples.png` y
`output/jenaro_ep2/images/cnn_gradcam_by_class.png`. Las imagenes de la raiz
conservan la evidencia previa del equipo.
El notebook y el script comparten `export_run` para que una nueva ejecucion
guarde juntos resumen, historial, metricas, figuras y modelo, sin mezclar
reportes de corridas distintas. La guia individual de la EP2 esta en
`presentation/jenaro_defensa_ep2.md`; `jenaro_defensa.md` pertenece a la EP1.

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
