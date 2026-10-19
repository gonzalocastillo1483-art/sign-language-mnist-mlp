# Evaluación y cierre de Jason EP2

La evaluación utiliza la ejecución de Jason en `output/jason_ep2/`, generada con el notebook completo y el código actualizado de Jenaro. Se mantienen arquitectura, semilla 42, particiones y presupuesto de 20 épocas. No se modificó la arquitectura ni se hizo una búsqueda basada en test.

## Resultados de la ejecución utilizada

| Dato | Resultado |
| --- | ---: |
| Imágenes de test | 7.172 |
| Aciertos / errores | 7.031 / 141 |
| Accuracy de test | 98,03 % |
| Precisión macro | 0,9804 |
| Recall macro | 0,9773 |
| F1 macro | 0,9786 |
| Tiempo registrado | 137,20 s |
| Parámetros | 243.352 |
| Brecha validación-test | 1,97 pp |

Frente al MLP histórico, la mejora de accuracy es 18,32 pp y la de F1 macro es 0,2025, usando 57,25 % menos parámetros. El registro original del MLP indica 31,450219 s y se muestra como 31,45 s. Los entornos difieren, por lo que los tiempos no constituyen una comparación controlada de velocidad.

## Trabajo incorporado

- Secciones 6 a 9 del informe: métricas, fórmulas, matriz, casos concretos, comparación, limitaciones y conclusiones.
- Actualización de las cifras e historial de la sección 5 para que correspondan a la ejecución seleccionada. Se conservan la arquitectura y las explicaciones de interpretabilidad de Jenaro.
- README y guion grupal coherentes con la ejecución de Jason. Las corridas previas permanecen identificadas como históricas.
- Presentación completa de diez diapositivas basada en la revisión de Jenaro, con curvas y mapas de la corrida actual, y actualización del bloque 7 a 10.
- Cuatro diapositivas individuales de Jason y guion con tiempos de 06:00 a 10:00 y respuestas de defensa.
- Predicciones individuales, matriz numérica, métricas por clase, conteos de errores y comparación exacta en CSV.
- Script de evaluación independiente que carga los pesos guardados y exporta la evidencia sin entrenar.

## Comprobaciones realizadas

El notebook ejecutó sus 16 celdas de código sin errores. Ambos JSON de métricas de la corrida son idénticos. Se cargó el modelo `.keras` y se volvió a predecir el test completo: accuracy, precisión, recall y F1 macro coinciden con los JSON, dentro de la tolerancia numérica. La matriz suma 7.172 imágenes y su diagonal suma 7.031 aciertos. Los principales errores son T como X (22), R como V (21), X como T (21), H como G (20), G como P (15) y K como Y (15).

Las cuatro pruebas existentes de `tests/test_cnn_interpretability.py` pasan. Las presentaciones superan las comprobaciones de estructura, geometría, fuentes, datos del gráfico editable y reimportación. Se renderizaron y revisaron las diez diapositivas. No se comprobó la apertura en PowerPoint.

El archivo `output/jason_ep2/requirements-lock.txt` registra las dependencias. `jason_evaluacion_verificada.json` conserva las huellas del modelo y del test. El modelo entrenado permanece local y excluido de Git.

## Reproducción

Desde la raíz, con el entorno preparado, para comprobar el modelo existente:

```powershell
.\.venv\Scripts\python.exe scripts\evaluar_jason_ep2.py --project-root .
```

El notebook fuente y `scripts/train_cnn_ep2.py` conservan la implementación de Jenaro. Para reconstruir todos los artefactos se puede ejecutar de nuevo ese notebook, definiendo `EP2_OUTPUT_DIR=output/jason_ep2`, o usar el script de entrenamiento con `--output-dir output/jason_ep2`. Una ejecución posterior puede cambiar métricas y tiempo; en ese caso deben actualizarse conjuntamente las cifras, imágenes y ejemplos del cierre.

Las corridas previas de 96,60 % y 96,81 % no se borran ni se mezclan con esta evaluación. La elección de la ejecución actual responde a verificar el flujo realizado por Jason, no a seleccionar un modelo ajustado con el test. Antes de la entrega resta la revisión grupal de la integración y el ensayo cronometrado.
