# Apoyo para la defensa de Jenaro

Duración orientativa: cuatro minutos, dentro de los diez minutos del equipo.
Leer los resultados actuales en `reports/jenaro_modelo_entrenamiento.md` y
abrir `images/jenaro_curvas_comparacion.png` antes de ensayar.

El objetivo de la exposición es demostrar comprensión del caso. Cada cifra
debe acompañarse de una interpretación y de sus límites. Un resultado bajo
también aporta si podemos explicar qué muestra y qué falta investigar.

## 1. Entrada y arquitectura: aproximadamente un minuto

Las imágenes tienen 28 por 28 píxeles: son 784 valores. Gonzalo ya los
normalizó entre 0 y 1 y los dejó como vectores. Por eso nuestra red recibe
784 entradas y no necesita volver a aplanar la imagen.

Una neurona calcula una suma de entradas multiplicadas por pesos, agrega
un sesgo y aplica una activación. En nuestras capas ocultas usamos ReLU,
que deja pasar los valores positivos y lleva los negativos a cero. Esa
no linealidad permite aprender relaciones más complejas entre los píxeles.

Comparamos tres capacidades de la actividad 1.4.4: una capa de 32 neuronas,
dos capas de 128 y 64, y tres capas de 512, 256 y 128. Queríamos comprobar
si aumentar capacidad mejoraba la validación y cuánto costaba entrenar.

## 2. Salida, pérdida y aprendizaje: aproximadamente un minuto

La salida tiene 24 neuronas, una por cada letra presente en estos datos.
Usamos softmax para obtener 24 valores que suman uno y elegimos el mayor.
El mapeo conserva la correspondencia con las letras: los índices consecutivos
de la red no son directamente los números originales del alfabeto.

Como en el notebook MNIST de clases, transformamos las etiquetas a one-hot:
un uno en la clase correcta y ceros en las demás. Usamos crossentropy para
medir qué tan bien las probabilidades coinciden con esa etiqueta.

Backpropagation calcula cómo afecta cada parámetro a la pérdida. Adam usa
esos gradientes para actualizar los pesos. Mantuvimos learning rate 0,001,
batch 128 y 20 épocas para comparar las arquitecturas bajo el mismo protocolo.

## 3. Comparación y resultados: aproximadamente un minuto

Mostrar la tabla real del informe y mencionar el nombre del modelo elegido,
su F1 macro de validación, accuracy y número de parámetros. Relacionar su
resultado con el de la red más pequeña y el tiempo de entrenamiento.

La selección se hizo con validación. Elegimos F1 macro porque cada letra
tiene el mismo peso en el promedio. La regla de desempate fue menor pérdida
de validación y después menor número de parámetros. El test oficial se
consultó solamente después de elegir el modelo.

Mostrar las curvas. Describir cómo cambian las pérdidas y señalar si hay
picos o una separación sostenida. No afirmar sobreajuste solo porque una
curva tiene un punto más alto. Comparar la accuracy de train y validación
calculadas con los mismos pesos finales.

## 4. Alcance y entrega: aproximadamente un minuto

Una opinión que podemos defender con esta ejecución es: el MLP resulta útil
como primera experiencia de clasificación de estas imágenes y permite ver
el efecto de la capacidad. La red grande obtuvo el mejor resultado de
validación bajo nuestro protocolo, pero ese resultado fue optimista: en
test acertó el 79,17%, es decir, se equivocó aproximadamente en una de cada
cinco imágenes. Por eso el 100% de validación no basta para calificar el
modelo como confiable para una aplicación de accesibilidad.

Podría servir como demostración educativa supervisada, donde se revisen
los errores. Para usarlo como evaluador automático del aprendizaje de una
persona, habría que comprobar que sus errores sean aceptables y probarlo
con imágenes representativas de ese uso. Ese criterio de aceptación y esas
pruebas todavía no están definidos; los datos actuales no permiten afirmar
que el producto esté listo.

La conclusión se limita a estas arquitecturas, partición, semilla y veinte
épocas. No basta una sola ejecución para decir que una arquitectura siempre
será mejor. Aumentar neuronas aumenta parámetros y costo, y no garantiza
generalizar mejor.

El modelo queda guardado junto con el mapeo de clases y el historial. Jason
utiliza las métricas y figuras finales para explicar errores y conclusiones.
El proyecto clasifica imágenes de letras ASL del dataset. Todavía no demuestra
reconocimiento en personas nuevas, fondos distintos o señas en movimiento.

## Preguntas para comprobar comprensión

| Pregunta | Idea que debe aparecer en la respuesta |
| --- | --- |
| ¿Qué aprende la red? | Pesos y sesgos. Las capas, épocas y learning rate son decisiones de configuración. |
| ¿Por qué 784 entradas? | 28 × 28 valores de píxeles. |
| ¿Por qué 24 salidas? | Hay 24 clases presentes; una neurona por clase. |
| ¿Por qué softmax y no una sola sigmoide? | Se debe escoger entre 24 clases, no resolver un problema binario. |
| ¿Qué hace one-hot? | Representa la clase con un vector; no normaliza los píxeles ni crea clases nuevas. |
| ¿Qué diferencia hay entre loss y accuracy? | La pérdida guía la actualización de pesos y usa probabilidades; accuracy cuenta clases acertadas. |
| ¿Qué es una época? | Un recorrido por todos los ejemplos de entrenamiento. |
| ¿Qué es el batch? | Ejemplos usados en una actualización. Aquí hay 172 lotes por época, incluido el último incompleto. |
| ¿Qué cambia con el learning rate? | La escala de la actualización. Muy alto puede causar oscilaciones; muy bajo puede retrasar el aprendizaje. |
| ¿Qué diferencia hay entre train, validación y test? | Train ajusta pesos; validación ayuda a seleccionar; test estima desempeño después de seleccionar. |
| ¿Más parámetros siempre mejora? | No. La tabla y las curvas permiten comparar beneficio, costo y posible sobreajuste. |
| ¿El modelo es bueno o malo? | Depende del propósito. Es útil para estudiar el flujo y comparar capacidades; el 79,17% de test y la diferencia con validación muestran límites para un uso automático confiable. |
| ¿Qué aprendemos del modelo pequeño? | Su pérdida sigue bajando y sus aciertos son menores. Puede faltar capacidad o más entrenamiento; este experimento no separa ambas causas. |
| ¿Podemos afirmar por qué baja el resultado en test? | Observamos una dificultad de generalización. Sobreajuste, semejanza entre imágenes de train/validación o diferencias entre conjuntos son hipótesis a investigar, no causas demostradas. |
| ¿Se pierde información al aplanar? | Los valores de los píxeles se conservan, pero una capa densa no incorpora filtros locales que aprovechen explícitamente la vecindad espacial. |
| ¿Por qué no usar test para escoger? | La elección se adaptaría a esos datos y debilitaría su función de evaluación final. |

La guía sirve para comprender y ensayar; no certifica que la defensa haya
sido realizada. La evaluación individual puede incluir preguntas sobre
el trabajo de Gonzalo y Jason.
