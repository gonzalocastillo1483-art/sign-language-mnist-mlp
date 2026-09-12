# Avance Gonzalo Castillo: carga y preprocesamiento

## Que hice

- Tome los CSV originales del dataset Sign Language MNIST desde `data/raw/`.
- Separe la columna `label` de los pixeles de cada imagen.
- Normalice los pixeles desde el rango 0-255 al rango 0-1.
- Prepare la entrada del MLP como vectores de 784 valores por imagen (28x28).
- Cree una division `train/validation` estratificada desde el set de entrenamiento oficial.
- Deje el set de test oficial separado para la evaluacion final.

## Resumen del dataset

- Dataset: Sign Language MNIST.
- Tipo de problema: clasificacion multiclase.
- Clases usadas: 24.
- Letras presentes: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y.
- Formato original: imagenes en escala de grises de 28x28 pixeles.
- Formato para el MLP: vector plano de 784 pixeles.

## Particiones preparadas

| Particion | Filas | Columnas de entrada | Labels |
| --- | ---: | ---: | ---: |
| Train | 21964 | 784 | 21964 |
| Validation | 5491 | 784 | 5491 |
| Test | 7172 | 784 | 7172 |

## Normalizacion

| Etapa | Minimo | Maximo |
| --- | ---: | ---: |
| Pixeles originales | 0 | 255 |
| Pixeles normalizados | 0.0000 | 1.0000 |

## Distribucion de clases

| Label original | Letra | Train | Test |
| --- | --- | ---: | ---: |
| 0 | A | 1126 | 331 |
| 1 | B | 1010 | 432 |
| 2 | C | 1144 | 310 |
| 3 | D | 1196 | 245 |
| 4 | E | 957 | 498 |
| 5 | F | 1204 | 247 |
| 6 | G | 1090 | 348 |
| 7 | H | 1013 | 436 |
| 8 | I | 1162 | 288 |
| 10 | K | 1114 | 331 |
| 11 | L | 1241 | 209 |
| 12 | M | 1055 | 394 |
| 13 | N | 1151 | 291 |
| 14 | O | 1196 | 246 |
| 15 | P | 1088 | 347 |
| 16 | Q | 1279 | 164 |
| 17 | R | 1294 | 144 |
| 18 | S | 1199 | 246 |
| 19 | T | 1186 | 248 |
| 20 | U | 1161 | 266 |
| 21 | V | 1082 | 346 |
| 22 | W | 1225 | 206 |
| 23 | X | 1164 | 267 |
| 24 | Y | 1118 | 332 |

## Evidencia generada

- `images/gonzalo_class_distribution.png`: distribucion de clases de entrenamiento.
- `images/gonzalo_sample_grid.png`: ejemplo visual de una imagen por clase.

## Nota para la defensa

Esta parte prepara los datos para que el MLP pueda entrenar. El punto clave es que una imagen
de 28x28 se aplana a 784 entradas numericas; eso permite usar capas
densas, pero tambien hace que el modelo pierda parte de la estructura espacial de la imagen.
