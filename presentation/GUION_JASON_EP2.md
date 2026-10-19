# Guion y defensa de Jason Aguilera

Tu bloque comprende las diapositivas 7 a 10, desde el minuto grupal 06:00 hasta el 10:00. Los tiempos son metas de ensayo. Las cifras corresponden a la ejecución guardada en `output/jason_ep2/`.

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
