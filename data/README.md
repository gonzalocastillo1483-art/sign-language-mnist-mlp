# Datos para ejecutar el proyecto

Los dos CSV originales están incluidos en el repositorio. Al clonar o descargar
el proyecto completo quedan en `data/raw/`, la ruta que utiliza el notebook 02.
No hace falta obtener ni extraer otro ZIP para ejecutar el proyecto.

| Archivo | Imágenes | Columnas | Tamaño en bytes |
| --- | ---: | ---: | ---: |
| `raw/sign_mnist_train.csv` | 27.455 | 785 | 83.281.065 |
| `raw/sign_mnist_test.csv` | 7.172 | 785 | 21.777.485 |

Cada fila contiene `label` y 784 intensidades de píxel. Las imágenes son de
28 × 28, en escala de grises, con 24 clases. El notebook crea la partición de
validación desde el CSV de entrenamiento y conserva el test oficial separado.

Ambos archivos coinciden byte por byte con los extraídos del archivo
`Sign Language MNIST.zip` utilizado por el equipo. No están normalizados ni
recortados: la preparación ocurre al ejecutar el código.

Huellas SHA-256 para comprobar la integridad:

```text
4c2897f19fab2b0ae2a7e4fa82e969043315d9f3a1a9cc0948b576bf1189a7e5  sign_mnist_train.csv
0e9d67bae23e67f40728e0b63bf15ad4bd5175947b8a9fac5dd9f17ce133c47b  sign_mnist_test.csv
```

Comprobación desde la raíz en PowerShell:

```powershell
Get-FileHash data/raw/sign_mnist_train.csv, data/raw/sign_mnist_test.csv -Algorithm SHA256
```

Las reglas de `.gitattributes` conservan los saltos de línea originales, incluso
al clonar en Windows. No se requiere Git LFS ni una descarga externa de datos.

La referencia de procedencia del dataset se completa en la sección 4 del
[informe técnico](../INFORME_TECNICO_BASE.md). La aprobación docente del dataset
ya está confirmada.
