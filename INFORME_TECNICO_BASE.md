# Clasificación de letras de ASL con un perceptrón multicapa

**Evaluación Parcial 1 de TLY1102**

**Integrantes:** Gonzalo Castillo, Jenaro Marín y Jason

**Docente:** Jose Rojas

**Sección:** 01V

**Fecha de entrega:** 15/09/2026

## Reparto para completar el informe

| Responsable | Secciones a cargo | Contenido |
| --- | --- | --- |
| **Gonzalo Castillo** | 1 a 4, 5.1, 6 y 7 | Problema, variante propia acordada, objetivos, KPIs, referencia del dataset, calidad, CRISP-DM y justificación de la preparación |
| **Jenaro Marín** | **8 y 9** | MLP, entrenamiento, comparación y elección del modelo |
| **Jason** | 5.2 y 5.3, 10 a 13 | interpretación de distribución e imágenes, métricas de test, matriz, aciertos, errores, limitaciones, impacto ético y conclusiones |
| **Todo el equipo** | Identificación, resumen, 14 y 15, anexos | identificación, síntesis, reproducción, fuentes y guía de defensa. Quedan la revisión de la ejecución final y el ensayo conjunto |

## Resumen

Desarrollamos un clasificador de imágenes de letras estáticas de ASL con el dataset Sign Language MNIST, aprobado por el profesor para este proyecto. El trabajo aplica los contenidos de perceptrón, redes Fully Connected, descenso del gradiente y evaluación de modelos vistos en clases. Su finalidad es comprender y explicar el flujo de aprendizaje supervisado, además de medir el desempeño obtenido.

El dataset contiene 34.627 imágenes de 28 × 28 píxeles y 24 clases. Utilizamos 21.964 imágenes para entrenamiento, 5.491 para validación y 7.172 del test oficial. La preparación divide las intensidades por 255, organiza 784 entradas por imagen y codifica las etiquetas en 24 posiciones mediante one-hot. Mantenemos las mismas particiones y un protocolo de 20 épocas, batch de 128 y Adam con learning rate 0,001 para comparar las arquitecturas.

Comparamos tres MLP con la misma preparación de datos y el mismo protocolo de entrenamiento. La configuración de 512, 256 y 128 neuronas ocultas obtuvo el mejor F1 macro de validación y se seleccionó antes de evaluar el test oficial. En test alcanzó **79,71% de accuracy y 0,7760 de F1 macro**, frente a 100% de accuracy en validación. La diferencia de 20,29 puntos porcentuales muestra que la validación interna fue optimista respecto de ese test. El modelo aprende patrones útiles, pero sus errores y la desigualdad entre letras limitan su uso como apoyo educativo fuera de las condiciones evaluadas.

La principal conclusión es que un resultado alto en validación no basta para asegurar utilidad fuera de ese conjunto. La evaluación por letra y los ejemplos de errores permiten fundamentar esa limitación. El repositorio reúne el código, los dos CSV originales, el notebook documentado y las evidencias del experimento. Este informe relaciona esas evidencias con las decisiones técnicas y organiza la reproducción y defensa del trabajo.

## 1 Descripción del problema de negocio

El usuario beneficiado se entiende como una persona que practica el alfabeto de ASL o como un docente que necesita revisar ejercicios básicos de reconocimiento visual. El modelo se plantea como una referencia técnica inicial para observar si una red densa puede distinguir letras estáticas a partir de imágenes ya preparadas.

Una aplicación para practicar el alfabeto de ASL podría utilizar un clasificador que asigne una letra a la imagen de una mano. Esto permitiría ofrecer retroalimentación durante ejercicios de reconocimiento. En esta primera etapa estudiamos si una red MLP puede resolver esa clasificación y qué errores habría que considerar antes de utilizarla con estudiantes.

La entrada es una imagen de 28 × 28 píxeles en escala de grises. La salida es una de las 24 letras presentes en el dataset. Es un problema de aprendizaje supervisado y clasificación multiclase: cada ejemplo incluye una etiqueta conocida y pertenece a una única clase.

El alcance comprende la clasificación de imágenes ya preparadas. No se implementa una aplicación con cámara, traducción de conversaciones ni reconocimiento de movimiento. Tampoco se evalúa lengua de señas chilena. La mejora del aprendizaje de usuarios sería un resultado de negocio relevante, pero este proyecto no realiza un estudio con personas que permita medirla.

### 1.1 Justificación del dataset

Sign Language MNIST permite practicar un flujo de imágenes con datos de tamaño manejable. Las imágenes tienen dimensiones homogéneas, etiquetas y un test oficial separado. Sus 784 píxeles se pueden utilizar como entrada de una red densa, lo que permite concentrarse en los fundamentos de MLP sin añadir una arquitectura fuera del alcance de esta experiencia.

La clasificación de letras visualmente parecidas también facilita analizar aciertos, confusiones y limitaciones. La aprobación del dataset está confirmada por el profesor.

### 1.2 Delimitación y variante del grupo

El experimento implementado utiliza las 24 clases disponibles, una partición estratificada con semilla 42 y la comparación de tres capacidades de MLP, manteniendo 20 épocas, batch de 128 y Adam con learning rate 0,001.

La variante propia del grupo consiste en mantener el problema multiclase con las 24 letras estáticas disponibles y comparar tres capacidades de MLP bajo un protocolo común. La diferencia respecto del problema general no está en crear un traductor completo ni en cambiar el dataset, sino en estudiar cómo varía el aprendizaje al aumentar la cantidad de capas y neuronas de una red Fully Connected sobre los mismos datos preparados.

Para que la comparación sea coherente, las tres arquitecturas usan la misma partición estratificada, la misma semilla 42, 20 épocas, batch de 128, Adam con learning rate 0,001 y las mismas métricas de evaluación. De esta forma, el efecto observado se relaciona principalmente con la capacidad del MLP y no con cambios simultáneos en los datos o en el protocolo de entrenamiento.

## 2 Objetivos del proyecto

Los objetivos se limitan al flujo técnico implementado: preparar datos, entrenar y comparar modelos MLP, evaluar resultados y documentar decisiones. No se plantea desplegar una aplicación real ni demostrar impacto educativo con usuarios, porque eso queda fuera del alcance de esta evaluación.

### 2.1 Objetivo general

Implementar y evaluar un modelo MLP para clasificar las 24 letras estáticas de Sign Language MNIST, justificando las decisiones de preparación, arquitectura y entrenamiento e interpretando sus resultados y limitaciones.

### 2.2 Objetivos específicos

- Describir las fuentes, la calidad y la distribución de los datos, identificando dificultades para la clasificación.
- Normalizar los píxeles, codificar las etiquetas y separar entrenamiento, validación y test con funciones distintas.
- Comparar tres configuraciones de capas densas bajo un protocolo común y seleccionar una mediante validación.
- Interpretar loss, accuracy, precision, recall, F1 y matriz de confusión, junto con ejemplos concretos de aciertos y errores.
- Documentar el procedimiento y organizar sus archivos para que otra persona pueda ejecutarlo y defender sus decisiones.

## 3 Definición de KPIs

Los KPIs se eligieron para describir el desempeño técnico del clasificador y para evitar depender de una sola cifra global. En un problema con 24 letras, accuracy permite resumir aciertos generales, pero puede ocultar diferencias entre clases. Por eso se incorporan métricas macro y por clase, que ayudan a revisar si algunas letras funcionan peor aunque el promedio total parezca aceptable.

Los siguientes indicadores técnicos permiten valorar la utilidad potencial del clasificador. No demuestran por sí solos una mejora del aprendizaje o de la accesibilidad de los usuarios.

| Indicador | Qué informa en este caso | Uso en el proyecto |
| --- | --- | --- |
| Accuracy | Proporción total de imágenes cuya letra se identifica correctamente | Resumir aciertos en validación y test |
| Precision por clase | De las imágenes predichas como una letra, cuántas corresponden realmente a ella | Detectar letras que el modelo predice en exceso |
| Recall por clase | De todas las imágenes reales de una letra, cuántas reconoce | Identificar letras que suelen quedar sin reconocer |
| F1 macro | Promedio del F1 de las 24 clases, dando el mismo peso a cada letra | Seleccionar la arquitectura en validación y complementar accuracy en test |
| Brecha de accuracy | Diferencia en puntos porcentuales entre validación y test | Describir cuánto cambia el desempeño entre conjuntos |
| Matriz de confusión | Relación entre letra real y letra predicha | Examinar dónde ocurren los errores |

Para una clase, precision = TP / (TP + FP), recall = TP / (TP + FN) y F1 = 2 × precision × recall / (precision + recall). TP son aciertos de esa clase, FP son predicciones incorrectamente asignadas a ella y FN son ejemplos de esa clase asignados a otra. Accuracy cuenta los aciertos globales. F1 macro promedia el F1 de cada clase, por lo que ayuda a detectar un desempeño desigual aunque algunas letras tengan más ejemplos.

El criterio experimental prioriza el mayor F1 macro de validación. No se fijó un umbral de aceptación para desplegar una aplicación real. Por ello, no corresponde declarar éxito comercial a partir de un porcentaje elegido después de observar el test. La pérdida se utiliza para interpretar el aprendizaje, aunque no es una medida directa de beneficio para el usuario.

## 4 Fuentes y comprensión de los datos

El dataset se trabaja en formato tabular. Cada fila representa una imagen y cada columna de píxel representa una intensidad de gris. Esto permite utilizar directamente los 784 valores como entrada de un MLP, sin requerir lectura de archivos de imagen separados.

La fuente utilizada es el archivo **Sign Language MNIST.zip**, del cual se extraen `sign_mnist_train.csv` y `sign_mnist_test.csv`. Ambos CSV originales están incluidos en `data/raw/` dentro del repositorio, con sus huellas en [la documentación del dataset](data/README.md). Cada fila contiene `label` y las columnas `pixel1` a `pixel784`. El archivo de test oficial se conserva separado durante el ajuste del modelo.

| Característica | Valor |
| --- | --- |
| Imágenes del entrenamiento oficial | 27.455 |
| Imágenes del test oficial | 7.172 |
| Total de imágenes | 34.627 |
| Dimensiones | 28 × 28 píxeles |
| Canales | Uno, escala de grises |
| Variables de entrada | 784 intensidades de píxel |
| Variable objetivo | `label` |
| Clases | 24 letras, A–I y K–Y |
| Rango original de intensidades | 0 a 255 |

Las etiquetas originales son 0–8 y 10–24. No están incluidas J ni Z, que requieren movimiento. No deben interpretarse los huecos en la numeración como clases observadas ni añadirse salidas sin ejemplos.

La procedencia del dataset corresponde a **Sign Language MNIST**, publicado en Kaggle por el usuario **tecperson/DataMunge**. La ficha del dataset lo describe como un reemplazo tipo MNIST para reconocimiento de gestos de mano y señala licencia **CC0: Public Domain**. En esta entrega se utiliza el archivo local `Sign Language MNIST.zip`, del cual se extrajeron los CSV conservados en `data/raw/`. Fuente consultada el 15/09/2026: https://www.kaggle.com/datasets/datamunge/sign-language-mnist.

## 5 Análisis exploratorio y calidad de los datos

### 5.1 Comprobaciones de estructura

Estas verificaciones permiten revisar que los archivos base tengan la estructura esperada antes de entrenar. Se comprueba tamaño, columnas, clases, nulos, duplicados exactos y rango de intensidades. Con esto se descartan problemas básicos de carga o formato, aunque no se puede asegurar que todas las etiquetas sean perfectas ni que las imágenes cubran todas las condiciones reales de uso.

La revisión de los CSV utilizados arroja los siguientes resultados. Los duplicados se cuentan como filas completas idénticas dentro de cada archivo.

| Comprobación | Entrenamiento oficial | Test oficial |
| --- | ---: | ---: |
| Filas | 27.455 | 7.172 |
| Columnas, incluida la etiqueta | 785 | 785 |
| Valores nulos | 0 | 0 |
| Filas duplicadas | 0 | 0 |
| Clases presentes | 24 | 24 |
| Intensidad mínima y máxima | 0 y 255 | 0 y 255 |

No se requiere imputación por valores faltantes en estos archivos. Estas verificaciones no descartan imágenes muy parecidas, errores de etiquetado ni dependencia entre fotografías de una misma persona. Tampoco demuestran que entrenamiento y test representen todas las condiciones de uso real.

Para reproducir estas comprobaciones desde la raíz del proyecto:

```python
import pandas as pd

for split in ("train", "test"):
    df = pd.read_csv(f"data/raw/sign_mnist_{split}.csv")
    pixels = df.drop(columns="label")
    print(split, df.shape)
    print("Nulos:", int(df.isna().sum().sum()))
    print("Filas duplicadas:", int(df.duplicated().sum()))
    print("Clases:", sorted(df["label"].unique()))
    print("Rango:", pixels.min().min(), pixels.max().max())
```

### 5.2 Distribución de las clases

En los datos de entrenamiento, cada letra tiene una cantidad parecida de imágenes. La letra E tiene 957 imágenes y la R tiene 1.294. En el test las cantidades cambian más: R tiene 144 imágenes y E tiene 498.

Por esta razón no basta con mirar un solo resultado general. También revisamos el resultado de cada letra. El F1 macro nos ayuda porque considera a todas las letras por igual, aunque una tenga menos imágenes que otra.

![Distribución de clases en el entrenamiento oficial](images/gonzalo_class_distribution.png)

*Figura 1. Cantidad de imágenes por letra antes de separar la validación. Fuente: análisis de Gonzalo, con detalle en [el reporte de preprocesamiento](reports/gonzalo_preprocessing_summary.md).* 

### 5.3 Exploración visual y variables relevantes

![Una imagen representativa por clase](images/gonzalo_sample_grid.png)

*Figura 2. Ejemplos de las 24 clases. Cada imagen muestra una letra, pero no representa todos los casos posibles.*

Las imágenes son de 28 × 28 píxeles y están en escala de grises. Para el modelo, cada píxel es un número. Al observar las imágenes se nota que la forma de los dedos, la posición de la mano y el contraste con el fondo pueden cambiar entre ejemplos. Algunas letras tienen formas parecidas, por lo que pueden ser más difíciles de separar.

Esto solo es una observación de las imágenes. No podemos afirmar que una postura o un fondo sea la causa de un error sin hacer otra prueba. La columna de letra correcta se usa para entrenar y revisar el resultado, pero no entra como dato al modelo.

## 6 Metodología CRISP DM

CRISP-DM se utiliza como guía para ordenar el proyecto desde la definición del caso hasta la evaluación. En esta entrega la etapa de despliegue no corresponde a producción, sino a dejar el repositorio, el informe y la presentación en condiciones de ser revisados y reproducidos.

| Etapa | Aplicación al proyecto | Evidencia |
| --- | --- | --- |
| Comprensión del negocio | Definir el apoyo educativo, la clasificación requerida y los KPIs | Secciones 1 a 3 |
| Comprensión de los datos | Examinar fuentes, clases, calidad y ejemplos | Secciones 4 y 5, figuras 1 y 2 |
| Preparación de los datos | Normalizar, remapear etiquetas y separar conjuntos | Sección 7 y `src/sign_mlp/data.py` |
| Modelado | Construir y comparar tres redes MLP | Secciones 8 y 9, notebook 02 |
| Evaluación | Interpretar métricas, curvas y errores respecto del objetivo | Secciones 9 a 12 |
| Despliegue | Plantear las condiciones para una entrega reproducible y un eventual uso | Secciones 13 y 14 |

No se realizó un despliegue en producción. La sexta etapa se aborda como planificación de la entrega y del posible uso futuro. Documentar el repositorio apoya esa etapa, pero no equivale a desplegar o validar una aplicación con usuarios.

## 7 Preparación y transformación de los datos

La preparación transforma los CSV originales en entradas numéricas compatibles con una red MLP. El objetivo es que los píxeles, etiquetas y particiones tengan un formato estable para entrenar, comparar y evaluar sin mezclar las funciones de cada conjunto de datos.

1. **Carga y separación de variables.** Se leen ambos CSV y se separa `label` de los 784 píxeles. Esto evita incorporar la respuesta como variable predictora.
2. **Conversión y normalización.** Se convierten los píxeles a `float32` y se dividen por 255. El rango queda entre 0 y 1. La división usa un valor fijo de la escala y no estima parámetros a partir del test.
3. **Formato de entrada.** Los CSV ya contienen los píxeles aplanados. Cada fila se utiliza como un vector de 784 valores. Para mostrar las imágenes se reconstruye la forma 28 × 28. El vector conserva los valores de los píxeles.
4. **Mapeo de etiquetas.** Se construyen índices consecutivos de 0 a 23 para las 24 letras. El helper actual consulta el vocabulario de etiquetas de ambos CSV para construir el mapeo. Esto no incorpora imágenes de test al ajuste de pesos ni usa su desempeño para elegir arquitectura. En una aplicación futura convendría fijar de antemano el vocabulario esperado.
5. **Partición estratificada.** Se reserva aproximadamente 20% de cada clase del entrenamiento oficial para validación, con semilla 42. La estratificación mantiene proporciones similares y evita que una clase quede ausente por una separación inadecuada.
6. **Codificación one-hot.** Cada etiqueta se representa con 24 posiciones, con un uno en la clase correcta y ceros en las demás. Esta codificación es coherente con softmax y `categorical_crossentropy`.

| Conjunto | Imágenes | Función |
| --- | ---: | --- |
| Entrenamiento | 21.964 | Ajustar pesos y sesgos |
| Validación | 5.491 | Comparar configuraciones y elegir el modelo |
| Test oficial | 7.172 | Evaluar el modelo después de seleccionarlo |

Se conserva la misma partición para las tres arquitecturas. No se aplicaron aumentación de datos, extracción de características con modelos preentrenados ni convoluciones. El procedimiento completo está en [el notebook principal](notebooks/02_sign_language_mnist_mlp_jenaro.ipynb) y [el módulo de datos](src/sign_mlp/data.py).

## 8 Diseño y justificación del MLP

### 8.1 Componentes de la red

Una neurona calcula una suma ponderada de entradas más un sesgo y luego aplica una función de activación. Los pesos determinan cuánto influye cada entrada. Al conectar varias capas, el MLP aprende combinaciones de los píxeles relevantes para distinguir letras.

La entrada tiene 784 valores. Las capas ocultas son Dense con ReLU, que permite introducir relaciones no lineales. La salida tiene 24 neuronas con softmax, una por clase. Se asigna la imagen a la clase con el mayor valor de salida. Que softmax produzca valores que suman uno no demuestra que estos estén calibrados como confianza para una aplicación real.

### 8.2 Arquitecturas comparadas

| Configuración | Capas ocultas | Cantidad de capas ocultas | Parámetros entrenables |
| --- | --- | ---: | ---: |
| A pequeña | 32 | 1 | 25.912 |
| B mediana | 128, 64 | 2 | 110.296 |
| C grande | 512, 256, 128 | 3 | 569.240 |

Las tres configuraciones siguen el tipo de comparación realizado en las actividades de clases. En este caso se adaptan a 784 entradas y 24 clases. A ofrece una referencia de menor capacidad; B y C permiten observar si un aumento conjunto de profundidad y anchura mejora la validación bajo el mismo presupuesto de épocas. Como se cambian ambos aspectos, el experimento no aísla el efecto del número de capas del efecto del número de neuronas.

Para una capa Dense, el número de parámetros es entradas × neuronas + neuronas, donde el último término corresponde a los sesgos. Por ejemplo, A tiene (784 × 32 + 32) + (32 × 24 + 24) = 25.912 parámetros.

### 8.3 Pérdida e hiperparámetros

| Decisión | Configuración | Justificación |
| --- | --- | --- |
| Activación oculta | ReLU | Introducir no linealidad con una función trabajada en clases |
| Activación de salida | Softmax | Representar la distribución entre 24 clases excluyentes |
| Función de pérdida | Categorical crossentropy | Comparar la distribución predicha con la etiqueta one-hot |
| Optimizador | Adam | Actualizar pesos a partir de los gradientes, conforme al ejemplo de Keras |
| Learning rate | 0,001 | Valor inicial común a las tres redes; no se afirma que sea el óptimo |
| Épocas | 20 | Fijar un presupuesto común y observar la evolución del aprendizaje |
| Batch size | 128 | Procesar grupos de ejemplos por actualización, igual para todos los modelos |
| Semillas | 42 para partición y entrenamiento | Registrar la aleatoriedad y facilitar la reproducción |
| Dropout | 0 | Comparar las redes densas sin añadir este factor al experimento |

La retropropagación calcula los gradientes de la pérdida respecto de los parámetros. Adam los utiliza para actualizar los pesos y sesgos. Una época recorre el entrenamiento completo y un batch corresponde a un grupo utilizado para una actualización.

La implementación se encuentra en [el constructor MLP](src/sign_mlp/model.py). El notebook pasa explícitamente `dropout=0` y `loss="categorical_crossentropy"`: estos argumentos son importantes porque los valores por defecto del helper conservan la configuración de la versión inicial del proyecto.

## 9 Entrenamiento y validación

Se realizó una corrida por arquitectura durante 20 épocas. Se registraron loss y accuracy por época en entrenamiento y validación. La comparación final utiliza los pesos de la última época. No se debe presentar esta experiencia como una búsqueda exhaustiva ni como una prueba de estabilidad entre muchas semillas.

El criterio de selección fue mayor F1 macro de validación. En un empate exacto, se considera menor pérdida de validación y después menor número de parámetros. El test oficial no intervino en esta elección.

### 9.1 Comparación de resultados

| Modelo | Accuracy train | Accuracy val | F1 macro val | Loss val | Tiempo fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| A pequeña | 67,35% | 65,60% | 0,6556 | 1,0748 | 10,8 s |
| B mediana | 98,34% | 97,80% | 0,9790 | 0,1178 | 10,4 s |
| C grande | 100,00% | 100,00% | 1,0000 | 0,0024 | 31,5 s |

Fuente: [tabla de comparación de validación](reports/jenaro_comparacion_validacion.csv). Los tiempos corresponden a la ejecución registrada y no son un benchmark general de las arquitecturas. Las pequeñas diferencias de tiempo entre A y B no permiten concluir que B siempre será más rápida.

Seleccionamos **C grande** por su F1 macro de validación. C utiliza aproximadamente 5,16 veces los parámetros de B. B ofrece un resultado de validación cercano con menor tamaño, pero no se evaluó una comparación de los tres modelos en test que permita afirmar que B generaliza mejor.

### 9.2 Interpretación de las curvas

![Curvas de entrenamiento y validación de las tres redes](images/jenaro_curvas_comparacion.png)

*Figura 3. Evolución de pérdida y accuracy durante 20 épocas. Los registros completos están en los archivos `reports/jenaro_historial_*.csv`.*

A conserva una pérdida mayor y sus curvas siguen mejorando al llegar a la época 20. Su resultado es compatible con aprendizaje insuficiente bajo este presupuesto. No permite distinguir por sí solo entre poca capacidad y necesidad de más épocas.

B reduce considerablemente la pérdida y mantiene resultados altos en validación. C aproxima ambas pérdidas a cero. En esta corrida no aparece una subida sostenida de la pérdida de validación mientras la de entrenamiento disminuye. Esto no garantiza un desempeño equivalente fuera de la validación interna.

Las pérdidas registradas durante `fit` acumulan el comportamiento de los lotes a lo largo de la época. Una evaluación posterior utiliza los pesos finales. Por ello, los valores de entrenamiento del historial y los calculados al finalizar pueden diferir ligeramente sin representar métricas del mismo instante.

### 9.3 Conclusión

La comparación responde a una pregunta concreta: cómo cambia el aprendizaje al
aumentar la capacidad del MLP con las mismas imágenes, partición e hiperparámetros
de entrenamiento. Con 20 épocas, A queda por debajo de B y C. El salto entre A y B
es de aproximadamente 32,20 puntos porcentuales de accuracy de validación. Entre
B y C, la diferencia es de 2,20 puntos, a cambio de pasar de 110.296 a 569.240
parámetros. Esto permite discutir el beneficio obtenido frente al tamaño de la red.

Elegimos C porque el criterio previo prioriza F1 macro de validación. Esa decisión
no significa que una red más grande siempre sea mejor ni que esté lista para uso
real. Una única corrida tampoco permite asegurar que el mismo orden se mantenga
con otras semillas. La interpretación debe respetar esas limitaciones.

El análisis final de generalización corresponde al aporte de Jason. Recibe el
modelo seleccionado y los registros de validación para interpretar el test, las
métricas por letra y los errores. Si el test resulta inferior, se debe explicar
ese resultado sin cambiar retrospectivamente el criterio que usamos para elegir C.

## 10 Evaluación del desempeño en test

Después de elegir el modelo C grande usando la validación, lo probamos con las 7.172 imágenes del test oficial. Estas imágenes se dejaron aparte durante el entrenamiento y la elección del modelo.

| Métrica | Resultado |
| --- | ---: |
| Accuracy | 79,71% |
| Precision macro | 0,7783 |
| Recall macro | 0,7859 |
| F1 macro | 0,7760 |
| Pérdida | 1,0483 |
| Diferencia entre validación y test | 20,29 puntos porcentuales |

La accuracy de 79,71% significa que el modelo acertó cerca de 8 de cada 10 imágenes del test. Precision y recall muestran, de forma general, si el modelo se equivoca al asignar una letra o si deja de reconocer letras que sí estaban presentes. El F1 macro junta ambas ideas y da la misma importancia a cada letra.

En validación el resultado fue 100%, pero en test bajó a 79,71%. Esto muestra que la validación fue más optimista que el test. No significa por sí solo que el modelo memorizó, ni permite decir exactamente cuál fue la causa. Para saberlo habría que hacer más pruebas.

## 11 Análisis de resultados y errores

### 11.1 Matriz de confusión

![Matriz de confusión del modelo seleccionado](images/jenaro_integracion_confusion_matrix.png)

*Figura 4. Las filas muestran la letra real y las columnas la letra que predijo el modelo.*

La diagonal principal muestra los aciertos. Las celdas que están fuera de esa diagonal muestran en qué letras se confundió el modelo. La matriz se debe leer junto con la cantidad de ejemplos de cada letra, porque una letra con más imágenes puede tener más errores en cantidad.

### 11.2 Resultado por letra

| Letra | Precision | Recall | F1 | Imágenes en test |
| --- | ---: | ---: | ---: | ---: |
| B | 0,99 | 0,95 | 0,97 | 432 |
| A | 0,88 | 1,00 | 0,94 | 331 |
| E | 0,89 | 1,00 | 0,94 | 498 |
| P | 0,94 | 0,89 | 0,91 | 347 |
| C | 0,87 | 0,93 | 0,90 | 310 |
| U | 0,72 | 0,55 | 0,63 | 266 |
| N | 0,71 | 0,54 | 0,61 | 291 |
| T | 0,60 | 0,59 | 0,59 | 248 |
| R | 0,49 | 0,72 | 0,58 | 144 |
| S | 0,58 | 0,48 | 0,52 | 246 |

La letra B tuvo un resultado alto. En cambio, S, R y T fueron las letras más difíciles de reconocer en esta selección. Por ejemplo, S tuvo F1 de 0,52. Esto confirma que el resultado general no cuenta toda la historia: algunas letras se reconocen mucho mejor que otras.

### 11.3 Ejemplos correctos

![Ejemplos de predicciones correctas](images/jenaro_integracion_correct_examples.png)

*Figura 5. Ejemplos en que la letra real y la letra predicha coinciden.*

Estos ejemplos muestran que el modelo aprendió patrones útiles para varias letras. De todos modos, que una imagen salga bien no quiere decir que esa letra siempre salga bien.

### 11.4 Ejemplos con errores

![Ejemplos de predicciones incorrectas](images/jenaro_integracion_incorrect_examples.png)

*Figura 6. Ejemplos del test en que la letra real y la predicción no coinciden.*

En los ejemplos aparecen casos como B predicha como U, N como A y F como C. Son ejemplos reales de la figura, pero no significan necesariamente que sean los errores más comunes. Las letras parecidas, la posición de la mano o el fondo podrían influir, pero por ahora son solo posibles explicaciones.



## 12 Limitaciones e impacto ético

### 12.1 Limitaciones técnicas

El modelo usa los 784 píxeles como una lista de números. Esto le permite usar la imagen, pero no aprovecha tan bien la cercanía entre píxeles como lo haría un modelo pensado especialmente para imágenes. Además, las fotos son pequeñas y muestran señas quietas.

Este trabajo no demuestra que el modelo funcione igual con videos, otras personas, distintos fondos o cambios de luz. Tampoco prueba que funcione con letras que requieren movimiento, como J y Z. Solo se hizo una corrida por cada tamaño de red, por lo que faltaría repetir pruebas para saber si los resultados se mantienen.

### 12.2 Uso responsable

Si un sistema educativo muestra una letra equivocada, podría confundir a quien está aprendiendo. Por eso no se debe presentar este modelo como una respuesta perfecta. Una versión futura debería mostrar sus límites y permitir comparar la respuesta con material confiable.

Tampoco evaluamos si el modelo funciona igual para todas las personas o condiciones de captura. Si en el futuro se usan fotos de usuarios, habría que pedir permiso, guardar la menor cantidad posible de datos y proteger esa información. Este proyecto es una práctica de clasificación, no una solución completa de accesibilidad.

## 13 Conclusiones y mejoras futuras

El proyecto permitió seguir todo el proceso: preparar datos, entrenar tres modelos, comparar sus resultados y revisar sus errores. El modelo C grande fue el mejor en validación, por eso se eligió antes de mirar el test.

En el test obtuvo 79,71% de accuracy y 0,7760 de F1 macro. Es un resultado que muestra que el modelo puede reconocer muchas letras, pero no todas con la misma calidad. La diferencia entre validación y test también muestra que un 100% en validación no basta para decir que el modelo funcionará igual en otros datos.

Como siguiente paso, podríamos cambiar una cosa a la vez, por ejemplo el número de épocas o el learning rate, y comparar el resultado. También sería útil repetir las pruebas con distintas semillas. Más adelante se podrían estudiar modelos para imágenes, como una CNN, o nuevas formas de aumentar los datos. Estas son ideas futuras y no forman parte de los resultados actuales.

## 14 Reproducción y organización de la entrega

La entrega se organiza como un proyecto reproducible: los datos se incluyen en el repositorio, el notebook principal contiene el flujo de trabajo y los reportes conservan las métricas y configuraciones. La presentación sintetiza las decisiones y resultados para la defensa de TLY1102, sección 01V, con el docente Jose Rojas.

### 14.1 Archivos y entorno

La ejecución de referencia está documentada con Python 3.13.6, TensorFlow 2.20.0 y Keras 3.15.1. Las dependencias completas están fijadas en [requirements-jenaro.txt](requirements-jenaro.txt). Las versiones, semillas, mapeo y huellas de los archivos se registran en [los metadatos de entrenamiento](reports/jenaro_entrenamiento.json). El equipo debe conservar una ejecución consistente de notebook, reportes y modelo.

```text
sign-language-mnist-mlp/
  INFORME_TECNICO_BASE.md
  README.md
  requirements-jenaro.txt
  data/raw/
    sign_mnist_train.csv
    sign_mnist_test.csv
  notebooks/
    02_sign_language_mnist_mlp_jenaro.ipynb
  src/sign_mlp/
    data.py
    model.py
  scripts/
    prepare_data.py
    gonzalo_preprocessing_report.py
  reports/
    jenaro_comparacion_validacion.csv
    jenaro_historial_*.csv
    jenaro_entrenamiento.json
    jenaro_integracion_test.json
    jenaro_integracion_classification_report.txt
  images/
  models/
    jenaro_mlp.keras
  presentation/
    guion.md
    base_presentacion_equipo.md
    archivos/Sign_Language_MNIST_base_final.pptx
```

**Los dos CSV originales están incluidos en el repositorio.** Al clonar o descargar el proyecto completo quedan en `data/raw/`, listos para las rutas del notebook. No hace falta compartir un ZIP del dataset por separado ni modificar rutas personales. Las huellas de los archivos se documentan en [data/README.md](data/README.md).

El modelo binario permanece fuera de Git porque se genera al ejecutar el notebook. No se requiere un modelo previo para entrenar desde cero. Si el equipo entrega además un modelo entrenado, deberá corresponder a los reportes de esa misma ejecución. Las dependencias de Python se instalan por separado. No se incluyen entornos virtuales, credenciales ni sesiones de Colab.

### 14.2 Pasos de ejecución

Para obtener una copia nueva, ejecutar en PowerShell desde la carpeta donde se desea guardar el proyecto:

```powershell
git clone https://github.com/gonzalocastillo1483-art/sign-language-mnist-mlp.git
cd sign-language-mnist-mlp
```

Si ya se dispone del proyecto, comenzar desde su raíz. Crear el entorno con Python 3.13.6, versión utilizada en la ejecución de referencia, e instalar las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-jenaro.txt
```

Los CSV ya están disponibles en `data/raw/`. No se necesita descargar el dataset por separado, extraerlo de nuevo ni modificar el código para apuntar a una carpeta de Descargas. Antes de entrenar se puede comprobar el entorno y la presencia de los datos:

```powershell
.\.venv\Scripts\python.exe -m pip check
Get-FileHash data/raw/sign_mnist_train.csv, data/raw/sign_mnist_test.csv -Algorithm SHA256
```

Las huellas deben coincidir con las publicadas en [data/README.md](data/README.md). Para abrir el notebook principal con el Python del entorno:

```powershell
.\.venv\Scripts\python.exe -m notebook notebooks\02_sign_language_mnist_mlp_jenaro.ipynb
```

Seleccionar el entorno correcto, reiniciar el kernel y ejecutar las celdas desde el principio. La ejecución entrena las tres redes y puede regenerar reportes, figuras y el modelo. Si se desea conservar una ejecución anterior, guardar sus resultados en una copia del proyecto antes de volver a entrenar. Las diferencias de entorno pueden producir variaciones numéricas; cualquier nueva corrida debe mantener consistentes sus resultados y explicaciones.

Como alternativa, se puede ejecutar el notebook completo desde terminal. Este comando guarda una copia con las salidas y conserva el archivo del notebook original, aunque el código sí regenera los artefactos del experimento en esa copia del proyecto:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute notebooks\02_sign_language_mnist_mlp_jenaro.ipynb --output 02_sign_language_mnist_mlp_ejecutado.ipynb --ExecutePreprocessor.timeout=1800
```

Al terminar, comprobar que ninguna celda haya registrado una excepción y que estén presentes la comparación de los tres modelos, las curvas, las métricas de test y los ejemplos. El archivo `models/jenaro_mlp.keras` se crea durante la ejecución. Si cambian los resultados, actualizar conjuntamente las cifras y su interpretación en el notebook, los reportes, el informe y la presentación.

### 14.3 Responsabilidades y defensa

| Integrante | Responsabilidad principal | Tiempo de exposición |
| --- | --- | ---: |
| Gonzalo | Problema, datos, EDA y preparación | 3 minutos |
| Jenaro | Arquitectura, entrenamiento y comparación | 4 minutos |
| Jason | Evaluación, errores, limitaciones y conclusiones | 3 minutos |

La presentación dura diez minutos y la evaluación de la defensa es individual. Cada integrante debe poder explicar también las partes de sus compañeros, debido a las preguntas cruzadas indicadas en la pauta. La [base de diapositivas](presentation/base_presentacion_equipo.md) y el [apoyo de Jenaro](presentation/jenaro_defensa.md) sirven para ensayar, contrastando siempre las cifras con los reportes actuales.

Gonzalo establece el problema y explica cómo las imágenes llegan preparadas al modelo. Jenaro retoma ese punto para describir el MLP y justificar su elección mediante validación. Jason recibe el modelo seleccionado y explica cuánto se sostiene su desempeño en test, cuáles son sus errores y qué implican para el caso. Esta secuencia conecta las tres intervenciones con un mismo procedimiento.

### 14.4 Comprobaciones de reproducción

Se comprobó la carga desde un clon nuevo con las dependencias disponibles. Los CSV conservan exactamente las huellas del ZIP original y el módulo de preparación obtiene las siguientes entradas:

| Comprobación | Resultado verificado |
| --- | --- |
| Datos incluidos | CSV de entrenamiento y test disponibles en `data/raw/` |
| Entrenamiento preparado | 21.964 filas y 784 variables |
| Validación preparada | 5.491 filas y 784 variables |
| Test preparado | 7.172 filas y 784 variables |
| Normalización | Valores entre 0 y 1 |
| Etiquetas remapeadas | 24 índices consecutivos, de 0 a 23 |
| Notebook publicado | Estructura y sintaxis válidas, 17 celdas de código con salidas guardadas y sin errores registrados |
| Referencias locales | Archivos enlazados y seis figuras disponibles |

Esta comprobación confirma la integridad de los insumos y la preparación desde el clon. No incluye una repetición del entrenamiento completo en ese clon ni sustituye el ensayo del equipo. La ejecución publicada aporta las salidas del experimento de referencia. El cierre de la entrega debe verificar que los resultados utilizados en la defensa correspondan a una misma ejecución.

### 14.5 Organización del ensayo

El primer ensayo consiste en recorrer las diez diapositivas respetando el reparto de tres, cuatro y tres minutos. Cada integrante debe explicar una decisión y una evidencia de su parte, sin limitarse a leer la pantalla. El segundo ensayo utiliza las preguntas del anexo B y permite que responda una persona distinta del responsable habitual de esa sección.

Para cerrar el ensayo, contrastar las cifras mencionadas con los reportes, corregir afirmaciones que no estén respaldadas y comprobar que la suma de las intervenciones no exceda diez minutos. El ensayo queda como actividad del equipo y no se registra como realizado hasta llevarlo a cabo.

## 15 Referencias y evidencias

Las referencias se organizan según su función: la pauta define lo que se evalúa, el material de clases aporta los fundamentos y los archivos del proyecto respaldan los resultados. Los materiales docentes conservan sus nombres de origen para facilitar su identificación. La ficha bibliográfica del dataset continúa asignada a Gonzalo en la sección 4.

### 15.1 Pauta y contenidos de clases

- **Pauta oficial:** `EP1_TLY1102_Instrucciones y Pauta PRESENTACIÓN_Estudiante.pdf`. Requisitos formales en páginas impresas 6 y 7, desarrollo y evaluación en 7 y 8, rúbrica en 10 y 11. Incluye un informe Markdown, notebook ejecutable, datos, organización del proyecto y defensa técnica.
- **Fundamentos de clases:** `1.1.1 El Perceptrón (2).pptx`, `1.1.2_Notebook_Construir_un_Perceptron_Estudiante_JR (2).ipynb`, `1.2.1_PPT_Redes_Fully_Connected (2).pdf`, `JR_1.2.4_Notebook_Entrenando_una_Red_FF_Estudiante (1).ipynb` y `1.3.1 Descenso del Gradiente.pptx`.
- **Diseño y evaluación de clases:** `1.3.4_Notebook_Loss_Gradiente_Estudiante.ipynb`, `1.4.1_PPT_Diseno_y_Evaluacion_Modelos (1).pptx`, `1.4.3_Diseño_Evaluacion_Modelos_Fully_Connected_Estudiante.ipynb` y `1.4.4_Notebook_Modelo_Keras_Metricas_Estudiante.ipynb`.
- **Dataset utilizado:** `Sign Language MNIST.zip`, del que provienen los dos CSV incluidos. La procedencia, enlace y condiciones de uso se detallan en la sección 4.

| Contenido de clases | Aplicación al proyecto |
| --- | --- |
| Perceptrón | Pesos, sesgo, combinación de entradas y activación |
| Redes Fully Connected | Capas densas y relación entre entradas, capas ocultas y salida |
| Pérdida y descenso del gradiente | Interpretación del error, gradientes y actualización de parámetros |
| Diseño y evaluación de modelos | Particiones, comparación de configuraciones e interpretación de curvas |
| Keras y métricas | One-hot, softmax, categorical crossentropy, accuracy y métricas por clase |

### 15.2 Evidencias del experimento

Los resultados de referencia corresponden a [la versión del experimento integrada por el equipo](https://github.com/gonzalocastillo1483-art/sign-language-mnist-mlp/tree/389909513ca2c3d29162d832fbbe74b70308a271). La incorporación posterior de los CSV y de la documentación no representa un nuevo entrenamiento. Las figuras provienen de los análisis del equipo.

| Evidencia | Archivo o ubicación | Qué permite comprobar |
| --- | --- | --- |
| Flujo completo | [Notebook 02](notebooks/02_sign_language_mnist_mlp_jenaro.ipynb) | Orden de preparación, entrenamiento y evaluación |
| Datos originales | [Documentación del dataset](data/README.md) | Archivos incluidos, tamaños y huellas de integridad |
| Preparación | [Reporte de Gonzalo](reports/gonzalo_preprocessing_summary.md) | Clases, particiones y normalización |
| Decisiones y entorno | [Metadatos de entrenamiento](reports/jenaro_entrenamiento.json) | Semillas, hiperparámetros, versiones y criterio de selección |
| Comparación de arquitecturas | [Tabla de validación](reports/jenaro_comparacion_validacion.csv) | Métricas de A, B y C con el mismo protocolo |
| Evolución del aprendizaje | [Historial A](reports/jenaro_historial_A_pequena.csv), [historial B](reports/jenaro_historial_B_mediana.csv) e [historial C](reports/jenaro_historial_C_grande.csv) | Loss y accuracy por época |
| Resultado global | [Métricas de test](reports/jenaro_integracion_test.json) | Accuracy 79,71%, F1 macro 0,7760 y brecha de 20,29 puntos |
| Diferencias entre letras | [Reporte por clase](reports/jenaro_integracion_classification_report.txt) | Precision, recall, F1 y soporte de las 24 letras |
| Evidencia visual | Figuras 1 a 6 de este informe, en `images/` | Distribución, ejemplos, curvas y matriz de confusión |

Los valores globales se toman de los JSON y CSV sin sustituirlos por el redondeo de la fila general del reporte por clase. Las métricas por letra se muestran con los dos decimales disponibles en ese reporte. Esta distinción evita confundir un redondeo de 0,80 con una accuracy exacta de 80%.

## Anexo A Correspondencia con la rúbrica

| Indicador | Ponderación | Desarrollo en el informe | Evidencia que debe acompañarlo |
| --- | ---: | --- | --- |
| Definición del problema y selección del dataset | 15% | Secciones 1 a 4 | Justificación y variante propia explícita |
| Preprocesamiento de datos | 15% | Secciones 5 a 7 | CSV, funciones de preparación y notebook |
| Implementación del MLP | 20% | Sección 8 | Arquitecturas, pérdida, activaciones e hiperparámetros justificados |
| Entrenamiento y validación | 15% | Sección 9 | Historiales, curvas interpretadas y criterio de selección |
| Análisis de resultados y errores | 20% | Secciones 10 a 13 | Métricas, matriz, aciertos, errores y limitaciones |
| Trazabilidad, documentación y presentación | 15% | Secciones 14 y 15 | Entorno, semillas, archivos coherentes y defensa individual |
| **Total** | **100%** | | |

La pauta exige explicación y evidencia. Una métrica alta no reemplaza la justificación del modelo ni la interpretación de sus errores.

## Anexo B Revisión antes de entregar

Esta lista organiza el cierre del informe y del paquete completo. Las casillas pendientes requieren una comprobación del equipo y no deben marcarse solo porque exista el nombre del archivo.

- [x] Describir el problema, los objetivos y los KPIs, distinguiendo alcance técnico y utilidad educativa.
- [x] Documentar fuentes disponibles, calidad, distribución de clases y ejemplos visuales.
- [x] Explicar CRISP-DM, normalización, etiquetas y particiones.
- [x] Presentar y justificar las tres arquitecturas, activaciones, pérdida y protocolo.
- [x] Incorporar curvas, comparación, métricas finales y análisis de aciertos y errores.
- [x] Discutir limitaciones, impacto ético y mejoras futuras sin presentarlas como resultados realizados.
- [x] Desarrollar la síntesis, las instrucciones de reproducción, las referencias disponibles y la guía de defensa común.
- [x] Completar la referencia de procedencia y condiciones de uso del dataset.
- [x] Incluir los dos CSV originales en el repositorio, conservando `data/raw/` y documentando sus huellas.
- [x] Verificar que los enlaces a archivos locales y las seis figuras del informe existen dentro del repositorio.

### Preguntas para la defensa

1. ¿Qué problema resuelve el modelo y qué usos quedan fuera de alcance?
2. ¿Cuál es la variante propia del equipo y dónde se refleja en el notebook?
3. ¿Por qué normalizar los píxeles y remapear las etiquetas?
4. ¿En qué se diferencian entrenamiento, validación y test?
5. ¿Cómo intervienen pesos, sesgos, ReLU, softmax y la pérdida en la predicción?
6. ¿Por qué se seleccionó C y qué costo tiene respecto de B?
7. ¿Qué aportan F1 macro y las métricas por clase que no muestra accuracy?
8. ¿Qué conclusiones permite la brecha de 20,29 puntos y cuáles requieren más evidencia?
9. ¿Qué error concreto del test pueden explicar como hipótesis y cómo lo comprobarían?
10. ¿Qué modificarían en un próximo experimento y cómo evitarían tomar todas las decisiones mirando el test?

### Respuestas de apoyo para las preguntas cruzadas

1. **Problema y alcance.** Clasificamos imágenes estáticas en una de 24 letras de ASL como ejercicio y referencia para un posible apoyo educativo. No evaluamos traducción de conversaciones, reconocimiento en video ni una aplicación con usuarios.
2. **Variante.** El experimento usa las 24 clases estáticas disponibles y compara tres capacidades de MLP con la misma partición, semilla 42 y protocolo de entrenamiento. La variante está en la comparación controlada de arquitecturas Fully Connected, no en una selección distinta de clases ni en un cambio del dataset.
3. **Preparación.** Dividir por 255 lleva las intensidades a 0–1. El remapeo resuelve los huecos de las etiquetas originales y permite representar 24 clases consecutivas. One-hot expresa la clase correcta en el formato requerido por la pérdida elegida.
4. **Particiones.** Entrenamiento ajusta los pesos, validación permite comparar configuraciones y test evalúa el modelo elegido. Utilizar el test para decidir repetidamente convertiría sus resultados en parte del proceso de ajuste.
5. **Componentes.** Los pesos y el sesgo forman una combinación de entradas, ReLU añade no linealidad y softmax produce la distribución entre clases. La pérdida compara esa distribución con la etiqueta y los gradientes orientan la actualización de los parámetros.
6. **Selección.** C obtuvo F1 macro de validación 1,0000 y ganó bajo el criterio establecido. Tiene 569.240 parámetros frente a 110.296 de B. El costo en tamaño debe mencionarse aunque el criterio de este experimento priorice F1 macro.
7. **Métricas.** Accuracy resume los aciertos totales. F1 macro da el mismo peso a cada letra, mientras precision y recall permiten diferenciar predicciones incorrectamente asignadas de letras que el modelo deja de reconocer.
8. **Brecha.** La caída de 100% a 79,71% muestra que el desempeño no se mantiene entre validación y test. No demuestra por sí sola memorización ni identifica la causa de los errores.
9. **Error concreto.** En la figura aparece una B predicha como U. La postura o el fondo podrían influir, pero son hipótesis. Habría que comparar ejemplos bajo condiciones controladas y medir el efecto para sostener una explicación causal.
10. **Próximo experimento.** Podríamos cambiar épocas o learning rate, mantener las demás condiciones y registrar la comparación. Las decisiones deben apoyarse en validación y reservar una evaluación independiente para el resultado final.
