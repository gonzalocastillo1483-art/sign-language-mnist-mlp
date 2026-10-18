# Guion de presentacion EP2

El guion esta pensado para explicar decisiones, no para leer todo el texto de
las diapositivas. Los numeros corresponden a la ejecucion de referencia.

## 00:00-00:35 - Diapositiva 1: caso y continuidad (Gonzalo)

"En la EP1 trabajamos con Sign Language MNIST y construimos un MLP para
clasificar 24 letras estaticas del alfabeto ASL. En esta segunda evaluacion
mantenemos exactamente el mismo problema, dataset y particiones, pero
reemplazamos el MLP por una CNN. Asi podemos comparar de manera directa cuanto
aporta una arquitectura disenada para imagenes."

## 00:35-01:30 - Diapositiva 2: continuidad y cambio de entrada (Gonzalo)

"No volvimos a hacer la limpieza de la EP1. Reutilizamos el mismo dataset ya
validado, las mismas 24 clases, las mismas particiones y la semilla 42. Esto es
importante porque permite atribuir las diferencias de rendimiento al cambio de
arquitectura. El MLP recibia cada imagen como un vector plano de 784 valores;
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
Las convoluciones buscan patrones locales y ReLU introduce no linealidad. Cada
bloque usa max pooling para reducir el tamano espacial y conservar las
activaciones mas importantes. Al final usamos una capa densa de 128 neuronas y
una salida softmax de 24 probabilidades."

## 03:45-04:50 - Diapositiva 5: entrenamiento y regularizacion (Jenaro)

"Entrenamos con Adam, learning rate 0,001, batch de 128 y hasta 20 epocas. La
perdida es sparse categorical crossentropy porque tenemos una etiqueta entera y
una sola clase correcta entre 24. Dropout apaga neuronas al azar durante el
entrenamiento para reducir sobreajuste. Tambien usamos EarlyStopping y reduccion
del learning rate si la validacion deja de mejorar."

## 04:50-05:50 - Diapositiva 6: curvas e interpretabilidad (Jenaro)

"Las curvas muestran que aumenta el accuracy y disminuye la perdida. Los mapas
de caracteristicas permiten ver que las primeras capas detectan bordes y la
silueta de la mano, mientras que las capas profundas concentran patrones mas
abstractos. Grad-CAM destaca las regiones de la imagen que mas influyen en la
prediccion."

## 05:50-06:00 - Cambio de expositor

Jenaro: "Jason presentara la evaluacion final y la comparacion con el MLP."

## 06:00-07:00 - Diapositiva 7: resultados de test (Jason)

"En las 7.172 imagenes de test, la CNN alcanzo 96,60% de accuracy, 0,9604 de
precision macro, 0,9616 de recall macro y 0,9595 de F1 macro. Usamos F1 macro
porque calcula el resultado por clase y luego las pondera por igual, de modo que
una clase con mas ejemplos no domina la evaluacion."

## 07:00-08:15 - Diapositiva 8: comparacion MLP versus CNN (Jason)

"El MLP obtuvo 79,71% de accuracy de test y la CNN 96,60%, una mejora de 16,89
puntos porcentuales. El F1 macro subio de 0,7760 a 0,9595. Ademas, la CNN uso
243.352 parametros frente a 569.240 del MLP. Su costo fue entrenar cerca de 10
segundos mas. La brecha entre validacion y test bajo de 20,29 a 3,40 puntos, por
lo que la CNN generaliza mucho mejor."

## 08:15-09:10 - Diapositiva 9: errores y matriz de confusion (Jason)

"La diagonal dominante de la matriz indica que la mayoria de las imagenes se
clasifican correctamente. Los errores mas frecuentes fueron T con X, G con Q y
X con T. Son senas con formas o posiciones de dedos parecidas. T fue la clase
con menor F1, con 0,7745, por lo que es el principal foco para mejorar."

## 09:10-10:00 - Diapositiva 10: conclusiones (Jason)

"Concluimos que la CNN es mas adecuada para este problema porque conserva la
estructura espacial y aprende patrones locales compartiendo pesos. Mejoro
fuertemente el rendimiento y la generalizacion usando menos parametros. Las
limitaciones son que solo reconoce senas estaticas y que no probamos imagenes
externas. Como trabajo futuro proponemos data augmentation, Batch Normalization
y validacion con camaras y personas distintas."

Jason debe terminar la ultima frase cuando el cronometro marque **10:00**.

## Respuestas cortas para la defensa

**¿Por que una CNN y no otro MLP?**  
Porque la CNN conserva la posicion de los pixeles, aprende patrones locales y
comparte filtros. El MLP aplana la imagen y pierde esa relacion espacial.

**¿Por que softmax?**  
Porque necesitamos 24 probabilidades que sumen 1 y elegir una sola clase.

**¿Por que no binary crossentropy?**  
Porque no son 24 decisiones independientes. Cada imagen pertenece a una sola
de 24 clases mutuamente excluyentes.

**¿Por que sparse categorical crossentropy?**  
Porque las etiquetas son indices enteros. Categorical crossentropy tambien
serviria si primero las convirtieramos a one-hot.

**¿Por que accuracy y F1 macro?**  
Accuracy resume el porcentaje total de aciertos. F1 macro revisa cada clase con
el mismo peso y combina precision y recall.

**¿Por que normalizar dividiendo por 255?**  
Porque 255 es la intensidad maxima de un pixel de 8 bits. Al dividir, el rango
pasa de 0-255 a 0-1 sin cambiar la informacion relativa.

**¿Por que pooling?**  
Reduce el tamano de los mapas, disminuye calculo y conserva las activaciones mas
fuertes de cada zona.

**¿Por que Dropout?**  
Ayuda a evitar que la red dependa demasiado de neuronas especificas y reduce el
riesgo de sobreajuste.

**¿La CNN es mejor en todo?**  
No. Fue mucho mejor en accuracy, F1, parametros y generalizacion, pero en esta
ejecucion necesito mas tiempo de entrenamiento.

**¿Por que la validacion llega a 100% pero el test no?**  
La validacion proviene del mismo archivo de entrenamiento y comparte mejor su
distribucion. El test oficial contiene ejemplos distintos y es una prueba mas
exigente de generalizacion.

**¿Que significa Grad-CAM?**  
Es un mapa de calor aproximado que muestra que regiones de la imagen influyeron
mas en la clase predicha.
