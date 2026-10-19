# Guion de presentacion EP2

El guion esta pensado para explicar decisiones, no para leer todo el texto de
las diapositivas. Los resultados del cierre corresponden a la ejecución de Jason en `output/jason_ep2/`.

## 00:00-00:35 - Diapositiva 1: caso y continuidad (Gonzalo)

"En la EP1 trabajamos con Sign Language MNIST y construimos un MLP para
clasificar 24 letras estaticas del alfabeto ASL. En esta segunda evaluacion
mantenemos exactamente el mismo problema, dataset y particiones, pero
reemplazamos el MLP por una CNN. Asi podemos comparar de manera directa cuanto
aporta una arquitectura disenada para imagenes."

## 00:35-01:30 - Diapositiva 2: continuidad y cambio de entrada (Gonzalo)

"No volvimos a hacer la limpieza de la EP1. Reutilizamos el mismo dataset ya
validado, las mismas 24 clases, las mismas particiones y la semilla 42. Esto permite comparar las configuraciones manteniendo los datos y las particiones. El MLP recibia cada imagen como un vector plano de 784 valores;
la CNN recibe la misma informacion conservando su forma espacial."

## 01:30-02:30 - Diapositiva 3: preparacion para la CNN (Gonzalo)

"Cada fila del CSV trae 784 valores porque 28 por 28 es 784. En el MLP usamos
ese vector plano. Para la CNN reconstruimos la imagen como un tensor de 28 por
28 por 1: alto, ancho y un canal de gris. Tambien dividimos cada pixel por 255.
Como la intensidad original va de 0 a 255, el resultado queda entre 0 y 1, lo
que estabiliza el entrenamiento. Esa reconstruccion tensorial es la adaptacion
principal de los datos para la EP2; no es una nueva limpieza. Conservamos
21.964 imagenes para entrenar, 5.491 para validar y 7.172 para test."

## 02:30-02:40 - Cambio de expositor

Gonzalo: "Ahora Jenaro explicara como la CNN procesa esos tensores."

## 02:40-03:45 - Diapositiva 4: arquitectura CNN (Jenaro)

"La red tiene tres bloques convolucionales con 32, 64 y 128 filtros de 3 por 3.
Los filtros comparten pesos en distintas posiciones y ReLU introduce no
linealidad. Las convoluciones mantienen alto y ancho; cada max pooling de 2 por
2 reduce la resolucion: 28, 14, 7 y 3. Flatten convierte 3 por 3 por 128 mapas
en 1.152 valores. Al final usamos Dense 128 y softmax con 24 probabilidades.
La arquitectura tiene 243.352 parametros entrenables."

## 03:45-04:50 - Diapositiva 5: entrenamiento y regularizacion (Jenaro)

"Entrenamos con Adam, learning rate 0,001, batch de 128 y hasta 20 epocas. La
perdida es sparse categorical crossentropy porque tenemos una etiqueta entera y
una sola clase correcta entre 24; no usamos one-hot. Dropout desactiva
activaciones al azar durante train y se desactiva al validar. Los callbacks
observan val_loss: EarlyStopping tiene paciencia 4 y restaura los mejores pesos;
ReduceLROnPlateau tiene paciencia 2 y divide la tasa por dos. En esta ejecución
se completaron 20 épocas, no hubo parada anticipada y se restauraron los pesos
de la época 19. El historial registra las tasas utilizadas. El entrenamiento
de esta corrida duró 137,20 segundos."

## 04:50-05:50 - Diapositiva 6: curvas e interpretabilidad (Jenaro)

"Las curvas muestran que aumenta el accuracy y disminuye la perdida. Los mapas
de caracteristicas muestran canales seleccionados de tres profundidades para
la misma imagen. Las primeras respuestas mantienen contornos y las profundas
son mas pequenas y selectivas. Grad-CAM combina mapas con gradientes de una
clase y los superpone a la entrada. Revisamos aciertos, errores y una muestra
por cada una de las 24 clases. Puede destacar mano, bordes o fondo; el mapa
es aproximado, esta normalizado por imagen y no demuestra causalidad."

El guion individual ampliado y las preguntas de Jenaro estan en
[`jenaro_defensa_ep2.md`](jenaro_defensa_ep2.md).

## 05:50-06:00 - Cambio de expositor

Jenaro: "Jason presentara la evaluacion final y la comparacion con el MLP."

## Bloque de Jason de 06:00 a 10:00

## Diapositiva 7 Métricas de test

**Tiempo: 06:00–07:00, 60 segundos.**

La evaluación utiliza las siete mil ciento setenta y dos imágenes del test oficial, separadas del entrenamiento. La CNN logró noventa y ocho coma cero tres por ciento de accuracy: siete mil treinta y un aciertos y ciento cuarenta y un errores. La precisión macro fue cero coma nueve ocho cero cuatro, el recall macro cero coma nueve siete siete tres y el F1 macro cero coma nueve siete ocho seis. Las métricas macro dan el mismo peso a cada letra, aunque tengan cantidades distintas de ejemplos. En la matriz, las filas son las clases reales y las columnas las predicciones. La diagonal concentra los aciertos. Verificamos las métricas usando nuevamente el modelo guardado, sin volver a entrenarlo.

## Diapositiva 8 Comparación MLP y CNN

**Tiempo: 07:00–08:15, 75 segundos.**

Ambos modelos utilizan los mismos datos, clases y partición con semilla cuarenta y dos. El MLP obtuvo setenta y nueve coma setenta y uno por ciento de accuracy y la CNN noventa y ocho coma cero tres. La mejora es de dieciocho coma treinta y dos puntos porcentuales. El F1 macro sube de cero coma siete siete seis cero a cero coma nueve siete ocho seis. La CNN utiliza aproximadamente doscientos cuarenta y tres mil parámetros frente a quinientos sesenta y nueve mil del MLP: cincuenta y siete coma veinticinco por ciento menos. La brecha entre validación y test baja de veinte coma veintinueve a uno coma noventa y siete puntos. Esto respalda mejor generalización al test oficial. El entrenamiento de esta CNN duró ciento treinta y siete segundos. Los tiempos pertenecen a entornos distintos y no permiten comparar velocidad de forma controlada. También cambian decisiones de regularización y selección de pesos.

## Diapositiva 9 Aciertos y errores

**Tiempo: 08:15–09:10, 55 segundos.**

Ahora mostramos una G correctamente reconocida y otra G confundida con P. También vemos una T clasificada como X y una X clasificada como T. Los conteos de toda la evaluación muestran veintidós casos de T como X, veintiuno de R como V y veintiuno de X como T. La clase T tiene el menor F1, cero coma nueve cero seis cinco, seguida por R y X. Esto permite identificar dónde concentrar una próxima revisión. Las siluetas y posiciones parecidas de los dedos son explicaciones posibles, aunque estas figuras y la matriz por sí solas no demuestran la causa de los errores.

## Diapositiva 10 Conclusiones

**Tiempo: 09:10–10:00, 50 segundos.**

Concluimos que esta CNN mejora el rendimiento sobre el test oficial y utiliza menos parámetros que el MLP. Su estructura aprovecha patrones locales de la imagen, lo que es coherente con la mejora observada. El alcance sigue siendo reconocer veinticuatro letras estáticas ASL. No evaluamos cámaras ni personas externas, y no reconocemos J, Z ni secuencias de video. Proponemos probar aumentación moderada y evaluar imágenes externas reservadas. Los ajustes deben elegirse con validación, conservando una evaluación independiente para comprobar si la mejora se mantiene. Muchas gracias.

## Respuestas para la defensa

**¿Qué mide accuracy?** La proporción de aciertos globales: 7.031 / 7.172 = 98,03 % aproximadamente.

**¿Precisión y recall son lo mismo?** No. Precisión pregunta cuántas predicciones de una letra son correctas. Recall pregunta cuántas imágenes reales de esa letra recupera el modelo.

**¿Por qué F1 macro?** Promedia el F1 de cada clase con el mismo peso. No es el F1 calculado a partir de los promedios de precisión y recall.

**¿Por qué 18,32 puntos porcentuales?** Es la diferencia entre 98,03 % y 79,71 %, usando los valores completos antes de redondear.

**¿Por qué ahora el accuracy difiere de 96,60 % o 96,81 %?** Son ejecuciones distintas del mismo diseño con entornos distintos. Esta evaluación utiliza los pesos, métricas y figuras de la ejecución de Jason. No se cambió la arquitectura para buscar un resultado mejor en test.

**¿De dónde salen los conteos?** De las predicciones del modelo guardado, exportadas en CSV. T como X tiene 22 casos; R como V y X como T tienen 21 cada uno.

**¿Por qué menos parámetros no implica entrenar más rápido?** El número de parámetros no mide todas las operaciones. Además, los tiempos registrados provienen de entornos distintos.

**¿Qué tiempo del MLP es correcto?** El registro original indica 31,450219 s, que redondea a 31,45 s. La tabla histórica usaba 31,50 s. La comparación exacta lee el registro original.

**¿Generaliza mejor?** Sobre este test oficial, sí: mejora accuracy y F1 y reduce la brecha con validación. No se ha demostrado el rendimiento con cámaras o personas externas.

**¿Un 100 % de validación garantiza generalización?** No. Debe contrastarse con datos separados. La causa de la diferencia puede requerir revisar distribución y similitud entre imágenes.

**¿Toda la mejora se debe a las convoluciones?** El diseño es coherente con la mejora, pero también cambian regularización y selección de pesos. No aislamos causalmente una sola decisión.

**¿Qué conserva el MLP al aplanar?** Conserva los valores de los píxeles, pero no incorpora explícitamente filtros locales compartidos ni vecindad espacial.

**¿Por qué no J y Z?** Requieren movimiento y no son clases presentes en este dataset estático. Tampoco evaluamos traducción de palabras ni lengua de señas chilena.

**¿Qué propones mejorar?** Revisar T, R y X; probar aumentación moderada en validación y comprobar el desempeño en imágenes externas reservadas. El test ya analizado no debe usarse para elegir repetidamente cambios.

**¿Qué es Grad-CAM?** Un mapa aproximado de regiones relacionadas con el puntaje de una clase. No prueba causalidad ni representa una medida de confianza.

**¿Por qué softmax y sparse categorical crossentropy?** Hay una sola clase correcta entre 24. Softmax normaliza las salidas y la pérdida sparse admite etiquetas enteras. El MLP usa one-hot con categorical crossentropy para el mismo objetivo multiclase.

## Ensayo

Empieza en 06:00 grupal. Cambia a la diapositiva 8 en 07:00, a la 9 en 08:15 y a la 10 en 09:10. Termina en 10:00. Ajusta el texto a tu ritmo y señala las figuras sin leer cada decimal. Conoce también el flujo general para preguntas cruzadas.

## Pregunta para otro grupo

¿Cómo comprobaron que la mejora se mantiene con datos distintos a los utilizados para elegir hiperparámetros?
