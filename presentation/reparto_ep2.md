# Reparto de trabajo EP2

El reparto se organiza por bloques de la pauta y mantiene una carga similar
para los tres integrantes. Todos deben conocer el flujo completo para las
preguntas cruzadas.

## Gonzalo Castillo: continuidad, datos y preparacion

- Explicar el problema resuelto en la EP1 y por que se mantiene el dataset.
- Definir la variante: clasificacion multiclase de 24 letras ASL.
- Describir tamano, formato, clases y calidad del dataset.
- Explicar el split de entrenamiento, validacion y test.
- Justificar la normalizacion de 0-255 a 0-1.
- Explicar por que la CNN recibe tensores `(28, 28, 1)` y no vectores de 784.
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

## Tiempos sugeridos

| Bloque | Responsable | Tiempo |
| --- | --- | ---: |
| Diapositivas 1-3 | Gonzalo | 2 min 40 s |
| Diapositivas 4-6 | Jenaro | 3 min 10 s |
| Diapositivas 7-10 | Jason | 3 min 30 s |
| Margen para cambios | Equipo | 40 s |

Total estimado: 10 minutos.
