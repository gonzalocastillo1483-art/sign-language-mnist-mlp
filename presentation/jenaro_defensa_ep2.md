# Defensa de Jenaro para la entrega 2

Mi bloque es arquitectura CNN, entrenamiento e interpretabilidad. Me corresponde
revisar y adaptar la base compartida del equipo, explicar sus decisiones y
comprobar sus evidencias. Esta guía corresponde a EP2;
`jenaro_defensa.md` conserva la preparación del MLP de EP1.

## Responsabilidad y tiempo

Según [el reparto EP2](reparto_ep2.md), mi parte comprende:

- Revisar la arquitectura en `src/sign_mlp/cnn.py`, el entrenamiento del script
  y del notebook CNN y su ejecución con semilla 42.
- Justificar convoluciones, pooling, capas densas, ReLU, softmax, Adam y pérdida.
- Explicar Dropout, EarlyStopping, ReduceLROnPlateau y las curvas.
- Generar y revisar Feature Maps y Grad-CAM, incluyendo cobertura de las clases.
- Completar las secciones 4 y 5 del informe y preparar las diapositivas 4, 5 y 6.

| Diapositiva | Tiempo del equipo | Duración de contenido |
| --- | --- | ---: |
| 4 Arquitectura | 02:40 a 03:45 | 65 s |
| 5 Entrenamiento | 03:45 a 04:50 | 65 s |
| 6 Interpretabilidad | 04:50 a 05:50 | 60 s |

Son **3 minutos y 10 segundos de contenido**. Entre 05:50 y 06:00 se entrega la
palabra a Jason. La pauta oficial fija un máximo de 10 minutos; el cierre en
10:00 exactos es una organización interna del equipo. La defensa es individual
y puede incluir preguntas sobre los bloques de Gonzalo y Jason.

## Guion para las tres diapositivas

La [presentación con mi bloque revisado](archivos/Sign_Language_MNIST_CNN_EP2_Jenaro.pptx)
incluye estas notas para las diapositivas 4, 5 y 6. Conserva las diez
diapositivas del equipo y sus cifras de referencia.

Los tiempos incluyen señalar las figuras y hacer pausas. Ensayar con cronómetro
y ajustar el ritmo; la cantidad de texto por sí sola no garantiza la duración.
Las cifras siguientes identifican la ejecución de referencia del repositorio.

### Diapositiva 4 en 65 segundos

“En mi bloque explico cómo la CNN procesa el tensor de 28 por 28 por un canal
que explicó Gonzalo.

La red usa tres convoluciones con 32, 64 y 128 filtros de tres por tres. Cada
filtro aprende pesos que se comparten en distintas posiciones: así detecta
patrones locales. ReLU deja pasar valores positivos y agrega no linealidad.

Después de cada convolución, max pooling toma el máximo de cada ventana de dos
por dos. El tamaño espacial pasa de 28 a 14, luego a siete y finalmente a tres.
Flatten convierte esos mapas en 1.152 valores para una capa densa de 128
neuronas. La salida softmax entrega 24 probabilidades, una por clase.

Esta configuración tiene 243.352 parámetros entrenables. Más filtros permiten
combinar más patrones; esta elección es una base experimental.”

### Diapositiva 5 en 65 segundos

“Para entrenar uso Adam con un learning rate inicial de 0,001, lotes de 128 y
un máximo de veinte épocas. Adam adapta las actualizaciones con el historial
de gradientes. La pérdida sparse categorical crossentropy corresponde a
una sola clase correcta y etiquetas enteras entre cero y 23; aquí no uso
one-hot.

Dropout desactiva aleatoriamente activaciones durante el entrenamiento: 25%
tras el segundo pooling y 40% antes de la salida. En validación se desactiva;
por eso puede superar a train.

Ambos callbacks observan la pérdida de validación. EarlyStopping espera cuatro
épocas sin mejora y restaura los mejores pesos. ReduceLROnPlateau espera dos
y reduce el learning rate a la mitad.

En la referencia se completaron veinte épocas en 41,41 segundos, sin parada
anticipada. La tasa bajó tres veces; el último cambio no se usó en otra época.
El 100% de validación todavía necesita contrastarse con test.”

### Diapositiva 6 en 60 segundos

“En las curvas observo pérdida decreciente y accuracy creciente. Validación
llega al 100% y train ronda el 99,4%. Eso no demuestra igual desempeño con
otras personas o cámaras.

En los Feature Maps comparo canales seleccionados de tres capas para la misma
imagen. En la primera todavía reconozco contornos; en la tercera aparecen
respuestas más localizadas y menor resolución. Un mapa activo no equivale a
una letra completa.

Grad-CAM responde otra pregunta: qué regiones contribuyen a la puntuación de
una clase. Comparo la imagen con el mapa y reviso tanto un acierto como un
error. Las activaciones pueden alcanzar bordes o fondo; no afirmo que siempre
mire solo la mano. El mapa es aproximado y no demuestra causalidad.

Jason presentará ahora las métricas finales, la comparación con el MLP y los
errores.”

## Mapa entre código y conceptos

| Dónde revisar | Qué debo poder explicar |
| --- | --- |
| [data.py](../src/sign_mlp/data.py), `prepare_sign_mnist_cnn_data` y `as_cnn_images` | Misma imagen normalizada, entrada `(N, 28, 28, 1)`; train ajusta pesos, validación guía decisiones y test evalúa al final. |
| [data.py](../src/sign_mlp/data.py), `make_label_mapping` y `remap_labels` | Los rótulos originales se convierten a índices consecutivos `0..23`. El índice de salida se devuelve a letra mediante el mapeo. |
| [cnn.py](../src/sign_mlp/cnn.py), `build_cnn` | Filtros 3×3, `padding="same"`, ReLU, pooling 2×2, Dropout, Flatten, Dense y softmax. |
| [cnn.py](../src/sign_mlp/cnn.py), `model.compile` | Adam, tasa inicial y pérdida sparse para etiquetas enteras; accuracy es una métrica distinta de la pérdida. |
| [cnn.py](../src/sign_mlp/cnn.py), `training_callbacks` | `val_loss`, paciencia 4 y restauración; reducción por factor 0,5, paciencia 2 y mínimo `1e-5`. |
| [train_cnn_ep2.py](../scripts/train_cnn_ep2.py), `main` y `export_run` | Semilla 42, configuración, `fit`, tiempo medido, evaluación y exportación de evidencias. |
| [evaluate.py](../src/sign_mlp/evaluate.py), `plot_history` | Curvas de pérdida y accuracy de train y validación. |
| [evaluate.py](../src/sign_mlp/evaluate.py), `feature_map_model` y `plot_feature_maps` | Activaciones intermedias de `conv_1`, `conv_2` y `conv_3`; canales seleccionados de una imagen. |
| [evaluate.py](../src/sign_mlp/evaluate.py), `gradcam_heatmap` y `plot_gradcam_examples` | Gradientes del logit, la puntuación previa a softmax; superposición sobre la imagen para la clase predicha. Distinguir etiqueta real, predicción y clase explicada. |
| [evaluate.py](../src/sign_mlp/evaluate.py), `plot_gradcam_by_class` | Una imagen de test por clase real; el mapa explica esa clase y permite revisar la cobertura de las 24 letras. |
| [Notebook CNN](../notebooks/03_sign_language_mnist_cnn_ep2.ipynb) | El mismo flujo explicado por etapas, ejecutable en orden. |

## Resultados de referencia y cómo leerlos

| Evidencia | Dato que acompaña la explicación |
| --- | --- |
| [cnn_metrics.json](../reports/cnn_metrics.json) | Referencia CLI: 20 épocas, 41,4098 s de `fit`, 243.352 parámetros, validación 100%, test 96,60% y F1 macro 0,9595. |
| [cnn_metrics_notebook.json](../reports/cnn_metrics_notebook.json) | Otra ejecución del notebook: 45,4922 s. Es otra medición de la misma CNN; no es el tiempo del MLP ni prueba una diferencia entre arquitecturas. |
| [cnn_history.csv](../reports/cnn_history.csv) | Contiene 20 filas, numeradas `0..19`; el último accuracy de train es 0,994263. Registra las tasas usadas: 0,001, luego 0,0005 y 0,00025. |
| [cnn_model_summary.txt](../reports/cnn_model_summary.txt) | 243.352 parámetros entrenables. El total 730.058 incluye 486.706 variables del optimizador Adam; no confundir ese estado con pesos de la red. |
| [Curvas](../images/cnn_training_curves.png) | Describir tendencia y diferencias entre train y validación antes de inferir causas. |
| [Feature Maps](../images/cnn_feature_maps.png) | Comparar profundidades para la misma entrada y distinguir resolución de cantidad de canales. |
| [Ejemplos Grad-CAM](../images/cnn_gradcam_examples.png) | Revisar un acierto y un error; las regiones activas pueden incluir fondo. Seis ejemplos por sí solos no cubren las 24 clases de la pauta. |

Una nueva ejecución puede cambiar tiempos y resultados. Antes de exponer,
confirmar que informe, notebook, PPT y figuras identifican la misma corrida.
La pauta pide activación de cada clase: comprobar la cobertura de las 24 letras
en la evidencia completa, además de los ejemplos breves de la diapositiva.

En la referencia, ReduceLROnPlateau redujo la tasa al cerrar las épocas 14, 17 y
20: `0,0005`, `0,00025` y `0,000125`. El historial registra la tasa antes de que
ese callback la actualice al final de la época. Por eso solo aparecen dos
reducciones usadas por épocas posteriores; la tercera ocurrió al terminar la
época 20 y no se utilizó en otra época.

## Validación local de la entrega 2

La ejecución local conserva la arquitectura de 243.352 parámetros, semilla 42,
particiones y presupuesto de 20 épocas. Usó TensorFlow 2.20.0 y Keras 3.15.1.
El entorno difiere de la referencia; los resultados no permiten atribuir una
diferencia a una biblioteca concreta ni comparar tiempos entre máquinas.
El guion oral anterior mantiene las cifras de referencia del equipo.

| Evidencia local | Qué revisar |
| --- | --- |
| [Métricas locales](../output/jenaro_ep2/reports/cnn_metrics.json) | 20 épocas en 127,95 s; validación 100%; test 96,81%; precision macro 0,9630; recall macro 0,9650; F1 macro 0,9630. |
| [Notebook ejecutado](../output/jenaro_ep2/03_sign_language_mnist_cnn_ep2_ejecutado.ipynb) | Flujo ejecutado y resultados de esta validación. |
| [Curvas locales](../output/jenaro_ep2/images/cnn_training_curves.png) | Tendencia de train y validación de esta corrida. |
| [Feature Maps locales](../output/jenaro_ep2/images/cnn_feature_maps.png) | Misma imagen junto a los primeros ocho canales de cada profundidad. |
| [Seis ejemplos Grad-CAM](../output/jenaro_ep2/images/cnn_gradcam_examples.png) | Imagen original y superposición; el mapa explica la clase predicha, indicada en el título. |
| [Grad-CAM de las 24 clases](../output/jenaro_ep2/images/cnn_gradcam_by_class.png) | Una imagen por clase real y un mapa que explica esa clase; cubre A a I y K a Y. |

Para practicar, comparar el acierto G con el error T predicha como X. En este
último caso, el mapa de ejemplos explica X, no T. Los mapas tienen intensidad
relativa por imagen: sus colores no sirven para comparar contribuciones
absolutas entre dos imágenes. La cobertura de 24 clases complementa los seis
ejemplos, sin demostrar que todas las imágenes de cada clase se expliquen igual.

## Preguntas prácticas para la defensa

| Pregunta | Respuesta que debo poder desarrollar sin leer |
| --- | --- |
| ¿Qué aprende un filtro y cuántos parámetros tiene `conv_1`? | Aprende nueve pesos por canal de entrada y un sesgo por filtro. `(3×3×1 + 1)×32 = 320`. Se comparte el mismo filtro en las posiciones espaciales. |
| ¿Cómo calculo los parámetros de `conv_2`? | Cada filtro recibe los 32 canales anteriores: `(3×3×32 + 1)×64 = 18.496`. No se cuentan solo nueve pesos por filtro. |
| ¿Qué dimensiones tiene cada bloque? | Conv conserva alto y ancho por `same`: `28×28×32`; pool `14×14×32`; conv `14×14×64`; pool `7×7×64`; conv `7×7×128`; pool `3×3×128`. |
| ¿Por qué el último pooling da tres y no 3,5? | La ventana y el desplazamiento son dos y se usa padding válido. Solo caben tres posiciones completas por eje de tamaño siete. |
| ¿Por qué Flatten da 1.152? | `3×3×128 = 1.152`. Reordena las activaciones; no aprende pesos. Dense 128 sí aprende `(1.152 + 1)×128 = 147.584` parámetros. |
| ¿Para qué sirve ReLU? | `max(0, x)` introduce no linealidad. Sin activaciones no lineales, componer capas lineales seguiría produciendo una transformación lineal. |
| ¿Qué hace max pooling? | Escoge el máximo local, reduce resolución y cálculo y puede dar tolerancia a pequeños desplazamientos. No conserva todo el detalle ni garantiza invariancia. |
| ¿Por qué filtros 3×3 y más canales al profundizar? | 3×3 capta vecindades pequeñas con pocos pesos. Más canales permiten combinar más patrones al bajar la resolución. Son decisiones razonables; no se probó que sean óptimas. |
| ¿Por qué softmax y sparse crossentropy? | Hay una clase correcta entre 24. Softmax produce probabilidades que suman uno y sparse recibe un entero `0..23`. Categorical crossentropy requeriría el vector one-hot; binary corresponde a otra formulación. |
| ¿Accuracy es la función que ajusta los pesos? | No. La pérdida usa las probabilidades y su gradiente guía el ajuste. Accuracy cuenta aciertos tras elegir una clase. |
| ¿Qué significa batch de 128? | Hasta 128 ejemplos por actualización de pesos. Con 21.964 imágenes hay 172 lotes por época, incluido el último incompleto. Una época recorre el conjunto de entrenamiento. |
| ¿Qué hace Adam? | Usa estimaciones de primer y segundo momento de los gradientes para adaptar las actualizaciones. El learning rate inicial controla su escala; Adam no evita por sí solo el sobreajuste. |
| ¿Dropout elimina neuronas del modelo guardado? | No. Durante entrenamiento anula activaciones al azar y reescala las restantes. En evaluación utiliza todas; no cambia el número de parámetros. |
| ¿Por qué validación puede ser mejor que train? | Train se mide mientras cambian los pesos y con Dropout activo; validación se calcula al final de la época sin Dropout. También depende de la dificultad del split; no demuestra una causa única. |
| ¿Qué vigilan los callbacks? | Ambos vigilan `val_loss`, no test. EarlyStopping espera cuatro épocas sin mejora y restaura mejores pesos. ReduceLROnPlateau espera dos y multiplica LR por 0,5, con mínimo `1e-5`. |
| ¿EarlyStopping detuvo esta referencia? | No: completó las 20 épocas. Estaba configurado y pudo restaurar mejores pesos, pero no hubo parada anticipada. |
| ¿Cuántas veces redujo la tasa ReduceLROnPlateau? | Tres, al cerrar las épocas 14, 17 y 20: a 0,0005, 0,00025 y 0,000125. Solo las primeras dos se usaron en épocas posteriores. La última ocurrió al finalizar el entrenamiento y no aparece como otra época en el historial. |
| ¿100% de validación significa que siempre acierta? | No. Es ese split interno. La referencia alcanza 96,60% en test y no demuestra desempeño con cámaras o personas externas. |
| ¿Una Feature Map es Grad-CAM? | No. La primera muestra activaciones de un canal intermedio. Grad-CAM combina mapas usando gradientes de una clase concreta y produce una localización aproximada. |
| ¿Por qué Grad-CAM usa logits en esta implementación? | Son las puntuaciones antes de softmax. Así se evita explicar una probabilidad saturada cercana a uno, cuyos gradientes pueden ser muy pequeños. El modelo conserva sus pesos y su salida softmax para clasificar. |
| ¿Grad-CAM prueba que el modelo entendió los dedos? | No. Muestra contribuciones a una puntuación a baja resolución; puede resaltar fondo o bordes y no demuestra causalidad. Hay que compararlo con la imagen y con los errores. |
| ¿El MLP pierde los valores al aplanar? | No: conserva los valores. La capa densa no incorpora explícitamente la vecindad y los filtros compartidos que usa la CNN. |
| ¿Qué significa la mejora frente al MLP? | En la referencia, accuracy de test pasa de 79,71% a 96,60%: 16,89 puntos porcentuales. Es evidencia de estas configuraciones y particiones, no una garantía para toda CNN. |

## Ensayo y comprobación personal

1. Abrir las diapositivas 4, 5 y 6 junto con las curvas y los mapas. Ubicar en el
   código cada decisión de la tabla y calcular a mano 320 parámetros y 1.152
   activaciones.
2. Hacer una pasada explicando sin cronómetro. Sustituir cualquier frase que no
   pueda responder con un ejemplo propio.
3. Hacer dos pasadas cronometradas: cambios de slide a 03:45 y 04:50, terminar
   contenido a 05:50 y dejar la transición prevista hasta 06:00. Si sobra texto,
   acortar ejemplos secundarios sin quitar convolución, entrenamiento ni mapas.
4. Señalar una región concreta de una Feature Map y de un Grad-CAM. Explicar
   qué se observa, qué clase se está explicando y qué no permite concluir.
5. Practicar con preguntas al azar, incluyendo datos y métricas de los otros
   integrantes. Responder en 20 a 30 segundos y apoyar la respuesta con una
   evidencia del proyecto.
6. Antes del ensayo completo, verificar cifras y cobertura de clases contra los
   archivos de la corrida que se presentará. No mezclar el tiempo CLI con el
   tiempo del notebook ni con una ejecución nueva.
