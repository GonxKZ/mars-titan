# Codificación histórica con recursos acotados

La edición histórica conserva sus vectores en Parquet por activo. La caché secundaria de gráficos puede desactivarse con `--no-cache-charts`, sin eliminar muestras, gráficos ni noticias de la salida. La caché de texto permanece disponible por contenido. La ruta predeterminada conserva ambos tipos de caché.

El materializador confirma cada activo mediante su recibo y sus huellas. `--max-new-assets` limita los activos nuevos intentados en una ejecución, sin contar los confirmados que se recuperan. `--min-free-disk-bytes` exige una reserva antes de cargar los modelos y comprueba el disco entre activos. Al alcanzar un límite, el progreso indica la causa y no se publica un manifiesto de corpus completo. Se puede reanudar con otro límite operativo. La política de caché forma parte de la identidad y no puede cambiar dentro de la misma edición.

Un corte durante un activo puede repetir la codificación de sus gráficos si no hay caché secundaria. Los activos confirmados se recuperan sin inferencia. La reserva de disco entre activos no es una cuota del sistema de archivos y no impide que otra aplicación consuma espacio durante una escritura. Los errores de entrada y de escritura siguen siendo errores, no ausencias admisibles.

`FrozenEncoders` utiliza pesos congelados y conserva FP32. `--cuda-memory-bytes` limita el asignador de Torch y `--min-free-cuda-bytes` comprueba la memoria libre antes de cargar los pesos. Este límite no incluye toda la memoria del contexto CUDA. La falta de CUDA o de la reserva requerida causa un error, sin pasar a CPU. `--text-batch-size` limita los fragmentos y `--image-batch-size` divide el lote de gráficos sin omitir el resto final. Los tamaños diferentes de los originales se registran en la identidad porque pueden cambiar el redondeo.

Las opciones pertenecen al comando existente y también están disponibles mediante `encode_corpus` y `FrozenEncoders`. Su ayuda se comprueba sin cargar modelos:

```bash
uv run python -m mars_titan.data.corpus_encoding --help
```

## Evidencia de capacidad y paridad

La auditoría de la edición anterior mide 7.995.239.328 bytes de Parquet y 10.616.950.784 bytes de caché para 1.816.369 muestras de 2.226 activos. La extrapolación simple a 17.076.024 ventanas históricas estima 75.164.737.259 bytes de Parquet y 65.149.949.885 bytes de caché. En ese momento había 135.054.024.704 bytes libres. La compresión y la proporción de noticias del histórico pueden diferir, por lo que estas cifras no sustituyen la medición de la nueva edición.

Los fixtures conservan los Parquet byte a byte al desactivar la caché de gráficos y verifican parada, recuperación, censo incompleto, valores no finitos y tipos de configuración. Una revisión independiente encontró que `False` y `0` se consideraban iguales al comparar diccionarios. La recuperación compara ahora la representación JSON canónica tanto en configuración como en manifiesto confirmado.

La comprobación CUDA utiliza MiniLM y ResNet18 congelados, dos textos y cinco imágenes sintéticas. Frente al recorrido anterior, la diferencia máxima es 5,96 × 10⁻⁸ en texto y cero en imágenes, con rtol=2 × 10⁻⁵ y atol=2 × 10⁻⁶ fijadas antes de ejecutar. El recorrido con lotes de dos alcanzó 575.889.408 bytes del asignador y 614.465.536 reservados. Había otra carga GPU activa. Los tiempos incluyen carga y calentamiento y no acreditan aceleración.

Esta comprobación permite preparar datos. No ejecuta aprendizaje ni evalúa capacidad predictiva. El [recibo](../../reports/data/historical-encoding-budgets-20261008.json) separa las pruebas, las estimaciones de disco y sus límites. El bloqueo de aprendizaje continúa hasta completar y verificar la edición histórica.
