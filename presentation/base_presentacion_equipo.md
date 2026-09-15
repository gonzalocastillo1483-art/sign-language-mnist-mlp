# Base de presentación — Sign Language MNIST con MLP

Presentación del equipo: Gonzalo, Jenaro y Jason. Duración: 10 minutos.
Base editable para Canva, con poco texto visible y explicaciones para ensayar.
Fuentes: versión del repositorio 389909513ca2c3d29162d832fbbe74b70308a271.
Las cifras de test vigentes son 79,71% de accuracy y 0,7760 de F1 macro.

## 1. Clasificar letras a partir de imágenes
**Gonzalo · 30 segundos**

**En pantalla**
- Sign Language MNIST: clasificación con redes Fully Connected.
- Objetivo: reconocer 24 letras estáticas de ASL.
- Gonzalo · Jenaro · Jason.

**Visual:** ejemplos reales del dataset, images/gonzalo_sample_grid.png.

**Notas para exponer:** Proponemos un caso educativo de reconocimiento de letras. No es un traductor completo de lengua de señas ni un sistema validado con cámara. El éxito se evalúa con accuracy, F1 macro y análisis de errores; no solo con el resultado de entrenamiento.

## 2. ¿Qué datos tenemos?
**Gonzalo · 60 segundos**

**En pantalla**
- Imágenes de 28 × 28 píxeles en escala de grises.
- 27.455 ejemplos de entrenamiento oficial y 7.172 de test.
- 24 clases: A–I y K–Y; J y Z no están incluidas.

**Visual:** distribución real de clases, images/gonzalo_class_distribution.png.

**Notas para exponer:** Cada fila contiene una etiqueta y 784 valores de píxeles. J y Z requieren movimiento y no están representadas en este conjunto estático. Las clases de entrenamiento tienen cantidades relativamente similares, aunque no idénticas. Esto no demuestra que cubramos todas las condiciones reales de captura.

## 3. Preparar datos y separar sus funciones
**Gonzalo · 90 segundos**

**En pantalla**
- Píxeles: dividir por 255 para obtener valores entre 0 y 1.
- Entrada: vector de 784 valores; salida: etiqueta one-hot de 24 clases.
- Entrenar: 21.964 · Validar: 5.491 · Probar: 7.172.

**Visual editable:** flujo datos → preparación → entrenamiento y validación → test final.

**Notas para exponer:** El flujo conecta comprensión del problema y datos, preparación, modelado y evaluación. La partición de validación sale del entrenamiento oficial; el test se conserva para la evaluación final. Normalizar evita trabajar con escalas de 0 a 255. Se remapean las etiquetas originales, que tienen huecos, a 24 índices consecutivos. Entrenamiento ajusta pesos; validación permite elegir configuración; test estima desempeño después de elegirla. No elegimos el modelo mirando el test.

## 4. ¿Cómo toma una decisión el MLP?
**Jenaro · 60 segundos**

**En pantalla**
- Entrada: 784 píxeles.
- Capas Dense con ReLU: combinan entradas y aprenden relaciones no lineales.
- Salida: 24 neuronas con softmax.

**Visual editable:** 784 → Dense + ReLU → … → Dense 24 + softmax.
Aclaración pequeña: las capas ocultas cambian entre configuraciones.

**Notas para exponer:** Una neurona calcula una suma ponderada más un sesgo y aplica una activación. ReLU permite aprender relaciones no lineales. Softmax entrega valores que suman uno; elegimos la clase con el mayor valor. Eso no garantiza que la confianza esté calibrada. Al vectorizar se conservan los píxeles, pero una red Dense no incorpora explícitamente la vecindad espacial como una convolución.

## 5. Entrenar: medir el error y ajustar los pesos
**Jenaro · 60 segundos**

**En pantalla**
- Pérdida: categorical crossentropy, con etiquetas one-hot.
- Optimizador: Adam · learning rate: 0,001.
- 20 épocas · batch de 128 · semilla 42.

**Visual editable:** predicción → pérdida → gradientes → actualización de pesos.

**Notas para exponer:** La pérdida mide qué tan desacertada es la distribución predicha respecto de la etiqueta. La retropropagación calcula gradientes y Adam los usa para actualizar los pesos. Una época recorre el conjunto de entrenamiento; el batch agrupa 128 ejemplos por actualización. Mantuvimos el mismo protocolo y particiones para comparar tres tamaños de MLP. No se usó dropout en estas configuraciones. Las métricas corresponden a la época 20.

## 6. Comparar tres tamaños de red
**Jenaro · 60 segundos**

**En pantalla — tabla editable**
| Modelo | Neuronas ocultas | Parámetros | Accuracy validación | F1 macro validación |
|---|---|---:|---:|---:|
| A | 32 | 25.912 | 65,60% | 0,6556 |
| B | 128 → 64 | 110.296 | 97,80% | 0,9790 |
| C | 512 → 256 → 128 | 569.240 | 100,00% | 1,0000 |

**Mensaje principal:** C fue seleccionado por F1 macro de validación, antes del test.

**Notas para exponer:** El criterio fue mayor F1 macro; los desempates consideran menor pérdida y luego menos parámetros. B obtiene un resultado cercano con aproximadamente cinco veces menos parámetros que C. Eso plantea un costo de complejidad, pero no demuestra que B sea mejor en test porque aquí no lo comparamos en ese conjunto. A rinde menos bajo este presupuesto de entrenamiento.

## 7. ¿Qué nos dicen las curvas?
**Jenaro · 60 segundos**

**En pantalla**
- A mantiene mayor pérdida y sigue aprendiendo al llegar a 20 épocas.
- B y C alcanzan mejores resultados de validación.
- Una buena validación necesita contrastarse con el test final.

**Visual dominante:** images/jenaro_curvas_comparacion.png, sin alterar los valores ni recortar ejes.

**Notas para exponer:** Loss y accuracy describen aspectos distintos. No observamos una subida sostenida de la pérdida de validación que permita afirmar sobreajuste a partir de estas curvas solamente. Como A todavía mejora al final, no podemos atribuir todo su resultado a pocas neuronas: también podría requerir más entrenamiento. La selección de C está justificada por validación, pero el test nos dirá cuánto se sostiene.

## 8. Evaluación final: la validación fue optimista
**Jason · 60 segundos**

**En pantalla**
- Accuracy en test: 79,71%.
- F1 macro: 0,7760.
- Diferencia validación–test: 20,29 puntos porcentuales.

**Visual editable:** dos barras, accuracy validación 100,00% y test 79,71%, eje de 0 a 100%.

**Notas para exponer:** El test contiene 7.172 imágenes. Precision macro es 0,7783, recall macro 0,7859 y pérdida 1,0483. Accuracy resume aciertos totales; F1 macro promedia por clase y da el mismo peso a cada una. El 100% de validación sí es un resultado medido, pero no garantiza generalización. La brecha muestra una limitación al evaluar imágenes del test; no prueba por sí sola memorización ni una causa específica.

## 9. ¿Dónde falla el modelo?
**Jason · 70 segundos**

**En pantalla**
- Menor F1 por clase: S ≈ 0,52; R ≈ 0,58; T ≈ 0,59.
- Ejemplos observados: B → U, N → A y F → C.
- Revisar semejanza visual, postura y fondo como posibles explicaciones.

**Visuales:** images/jenaro_integracion_confusion_matrix.png y images/jenaro_integracion_incorrect_examples.png.
Priorizar legibilidad: matriz como visual principal y ejemplos como apoyo; conservar archivos originales para ampliar durante el ensayo.

**Notas para exponer:** La diagonal de la matriz representa aciertos; fuera de ella aparecen confusiones. Los ejemplos mencionados existen en la figura, pero no deben presentarse como las confusiones más frecuentes sin contarlas. Algunas posturas o fondos podrían influir; son hipótesis, no causas demostradas. B tiene F1 cercano a 0,97, lo que muestra que el rendimiento no es uniforme entre letras.

## 10. Conclusión: útil como ejercicio, con límites claros
**Jason · 50 segundos**

**En pantalla**
- El MLP aprendió a clasificar letras, con desempeño desigual entre clases.
- El resultado de test limita la confianza que podemos poner en validación.
- Próximo paso: contrastar hipótesis cambiando una variable a la vez.

**Notas para exponer:** El trabajo permite explicar datos, arquitectura, pérdida, optimización y evaluación, que son contenidos de clases. No podemos afirmar que esté listo para reconocer señas en situaciones reales. Como siguiente experimento, podríamos variar épocas o learning rate con el mismo protocolo, registrar resultados y reservar una evaluación independiente para la decisión final. Más complejidad no garantiza mejores resultados fuera del entrenamiento.

## Organización del ensayo

| Integrante | Diapositivas | Tiempo |
|---|---|---|
| Gonzalo | 1–3 | 3 minutos |
| Jenaro | 4–7 | 4 minutos |
| Jason | 8–10 | 3 minutos |

Preguntas para ensayar:
- ¿Por qué separar entrenamiento, validación y test?
- ¿Por qué usar ReLU, softmax y categorical crossentropy?
- ¿En qué se diferencian loss, accuracy y F1 macro?
- ¿Qué evidencia permite afirmar que C fue la mejor configuración de validación?
- ¿Por qué 100% de validación no basta para declarar que el modelo es bueno?
- ¿Qué conclusiones están respaldadas y cuáles son hipótesis?

## Fuentes y control de cifras

Repositorio: https://github.com/gonzalocastillo1483-art/sign-language-mnist-mlp/tree/389909513ca2c3d29162d832fbbe74b70308a271
- reports/jenaro_comparacion_validacion.csv: comparación de configuraciones.
- reports/jenaro_integracion_test.json: métricas globales vigentes.
- reports/jenaro_integracion_classification_report.txt: métricas por letra.
- notebooks/02_sign_language_mnist_mlp_jenaro.ipynb: procedimiento e interpretaciones.
- images/: figuras originales del equipo.
- presentation/guion.md: distribución de responsabilidades y tiempo.

Criterio de contenido: pauta EP1 TLY1102 y material de clases aportado por el equipo, especialmente el notebook 1.4.4 sobre Keras y métricas. No se incorporan técnicas nuevas como parte del modelo realizado.
La cifra antigua de test 79,17% quedó reemplazada por 79,71% en los resultados actuales; no mezclar versiones.
