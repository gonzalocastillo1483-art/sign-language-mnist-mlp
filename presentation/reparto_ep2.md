# Reparto de trabajo EP2

El reparto se organiza por bloques de la pauta y mantiene una carga similar
para los tres integrantes. Todos deben conocer el flujo completo para las
preguntas cruzadas.

## Gonzalo Castillo: continuidad y adaptacion MLP-CNN

- Explicar brevemente el problema de la EP1 y por que se reutiliza el dataset.
- Justificar la continuidad de la variante multiclase de 24 letras ASL y la
  semilla 42, tal como exige el criterio de variante propia.
- Aclarar que no se repite la limpieza ni el EDA completo de la EP1: se
  reutilizan los datos ya validados y solo se verifican las condiciones
  necesarias para ejecutar la CNN.
- Explicar que se conservan las mismas particiones para que la comparacion con
  el MLP sea objetiva.
- Justificar la normalizacion de 0-255 a 0-1.
- Explicar por que la CNN recibe tensores `(28, 28, 1)` y no vectores de 784.
- Comparar conceptualmente el MLP, que recibe una entrada plana, con la CNN,
  que aprovecha vecindad espacial, filtros compartidos y pooling.
- Redactar las secciones 1 a 3 del informe.
- Presentar las diapositivas 1, 2 y 3.

## Jenaro Marin: arquitectura, entrenamiento e interpretabilidad

- Explicar y justificar las capas convolucionales, pooling y densas.
- Justificar ReLU, softmax, Adam y `sparse_categorical_crossentropy`.
- Ejecutar y revisar el entrenamiento con semilla 42.
- Interpretar las curvas de perdida y accuracy.
- Explicar Dropout, EarlyStopping y ReduceLROnPlateau.
- Generar e interpretar mapas de caracteristicas y Grad-CAM.
- Redactar las secciones 4 y 5 del informe.
- Presentar las diapositivas 4, 5 y 6.

## Jason Aguilera: evaluacion, comparacion y conclusiones

- Reportar accuracy, precision, recall y F1 macro de test.
- Interpretar la matriz de confusion y los errores mas frecuentes.
- Seleccionar predicciones correctas e incorrectas representativas.
- Comparar MLP y CNN en metricas, parametros, tiempo y generalizacion.
- Explicar limitaciones y proponer mejoras.
- Redactar las secciones 6 a 9 del informe.
- Presentar las diapositivas 7, 8, 9 y 10.

## Responsabilidades compartidas

- Ejecutar el notebook completo antes de entregar.
- Comprobar que PPT, informe, README y notebook muestren los mismos numeros.
- Ensayar una presentacion total de 10 minutos.
- Preparar preguntas para otro grupo.
- Practicar preguntas cruzadas: cualquier integrante puede recibir preguntas
  sobre arquitectura, perdida, metricas, errores o comparacion.

## Cronograma obligatorio de 10 minutos

El docente utilizara cronometro. Los cambios de expositor forman parte del
tiempo y no se deja un margen separado.

| Tiempo acumulado | Diapositiva | Responsable | Duracion |
| --- | --- | --- | ---: |
| 00:00-00:35 | 1. Caso y continuidad | Gonzalo | 35 s |
| 00:35-01:30 | 2. Continuidad MLP-CNN | Gonzalo | 55 s |
| 01:30-02:30 | 3. Adaptacion tensorial | Gonzalo | 60 s |
| 02:30-02:40 | Cambio a Jenaro | Gonzalo/Jenaro | 10 s |
| 02:40-03:45 | 4. Arquitectura CNN | Jenaro | 65 s |
| 03:45-04:50 | 5. Entrenamiento | Jenaro | 65 s |
| 04:50-05:50 | 6. Feature Maps y Grad-CAM | Jenaro | 60 s |
| 05:50-06:00 | Cambio a Jason | Jenaro/Jason | 10 s |
| 06:00-07:00 | 7. Metricas de test | Jason | 60 s |
| 07:00-08:15 | 8. Comparacion MLP-CNN | Jason | 75 s |
| 08:15-09:10 | 9. Analisis de errores | Jason | 55 s |
| 09:10-10:00 | 10. Conclusiones | Jason | 50 s |

**Total obligatorio: 10:00 exactos.**

Puntos de control durante el ensayo:

- Gonzalo debe entregar la palabra a los **02:30**.
- Jenaro debe terminar su contenido a los **05:50**.
- Jason debe comenzar a los **06:00** y cerrar a los **10:00**.
- Si alguien se atrasa, debe acortar ejemplos secundarios; no se eliminan la
  comparacion MLP-CNN, Feature Maps, metricas ni conclusiones.
