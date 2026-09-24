# Lectura de episodios y evaluación sin archivos intermedios

La comparación del 24 de septiembre de 2026 conserva dos cambios. La lectura de episodios reutiliza un `ParquetFile` de Arrow mientras recorre las cohortes del mismo bloque. La evaluación por época acumula los errores sin construir tablas de predicciones cuando no hay un destino Parquet. El cálculo del modelo, los pesos, las etiquetas y las reducciones de errores siguen el mismo recorrido.

El lector conserva un solo archivo abierto. Lo cierra al cambiar de bloque o cerrar la fuente. Sigue comprobando tamaño, huella inicial y firma del archivo antes de cada acceso, también cuando la cohorte ya está en caché. Los límites de dos grupos y del presupuesto de bytes no cambian. Las tablas leídas conservan sus buffers después de cerrar el archivo.

## Medidas y paridad

Se compararon los dos archivos originales de `4a5bc69` con las versiones modificadas, usando el mismo script y los mismos datos sintéticos. Cada caso tiene cinco procesos independientes y un calentamiento por proceso. Los hilos numéricos se limitaron a uno, las lecturas Arrow usan `use_threads=False` y la simulación usa un trabajador. La preparación de archivos se ejecuta en otro proceso. El pico de RAM procede de `VmHWM` después de `exec`, una aproximación asíncrona de Linux que evita heredar el máximo anterior del proceso lanzador.

| Recorrido | Referencia, ms | Modificado, ms | Cambio de mediana |
| --- | ---: | ---: | ---: |
| Episodios, 4 activos y 32 cohortes | 30,559 ± 2,042 | 20,558 ± 0,533 | −32,7 % |
| Episodios, 128 activos y 192 cohortes | 778,870 ± 12,906 | 720,610 ± 15,210 | −7,5 % |
| Evaluación sin destino, 1.024 filas | 1,023 ± 0,026 | 0,584 ± 0,031 | −42,9 % |
| Evaluación sin destino, 65.536 filas | 60,517 ± 2,028 | 32,482 ± 0,656 | −46,3 % |
| Evaluación guardada, 1.024 filas | 4,031 ± 0,357 | 4,013 ± 0,387 | −0,4 % |
| Evaluación guardada, 65.536 filas | 95,384 ± 2,580 | 95,845 ± 1,693 | +0,5 % |

La tabla recoge mediana y desviación muestral entre procesos. El recorrido de episodios incluye lectura de todas las modalidades, cálculo de una referencia fija y simulación contable completa. Sus contextos contienen 8 y 64 sesiones, respectivamente. La evaluación usa un modelo lineal de 16 entradas, lotes de 512 y datos analíticos residentes en CPU. Es un diagnóstico de E/S y acumulación, sin entrenamiento ni medición del rendimiento CUDA.

En el caso de 128 activos, el tiempo total de proceso pasó de 1,936 a 1,820 s de mediana y el pico de RAM de 153,74 a 153,98 MiB. En la evaluación sin destino de 65.536 filas, el tiempo total pasó de 1,678 a 1,614 s y el pico de 591,14 a 584,44 MiB. El tiempo total incluye importaciones y calentamiento. La evaluación guardada mantiene sus tiempos dentro de la dispersión observada.

El perfil separado del recorrido de episodios registra 193 aperturas Parquet en la referencia y 13 después del cambio, correspondientes a 12 bloques y un archivo contable. En la evaluación sin destino, las 128 llamadas a `SessionErrors.update` se conservan y pasan a ocupar 56,6 de los 60,5 ms del generador instrumentado. Esos tiempos incluyen el perfilador y no sustituyen las medidas de la tabla.

Las huellas de las entradas y los resultados contables de la simulación coincidieron exactamente. También coincidieron todas las métricas numéricas de evaluación y el SHA-256 de las predicciones guardadas. Los tiempos y los contadores de recursos quedan fuera de esa comparación numérica. Las predicciones conservan el mismo orden, esquema y columnas.

El [informe de medidas](../../reports/resources/evaluation-io-benchmark.json) contiene las repeticiones, perfiles separados, bytes, caudal, latencias de cohortes, versiones y huellas de código. La [verificación](../../reports/resources/evaluation-io-quality.json) recoge pruebas de interrupción y recuperación, límites de memoria, cierre de lectores, archivos alterados, paridad de métricas y mutaciones dirigidas.

Pasaron 34 pruebas específicas y se detectaron las cinco mutaciones ejecutadas. La cobertura de líneas ejecutables es del 97,1 % en `_group`, del 100 % en `close` y del 96,3 % en `_evaluate`, con su cierre `tables` al 100 %. El informe conserva la complejidad, la convención CRAP y el alcance de cobertura. Estas comprobaciones no equivalen a ejecutar la suite completa de entrenamiento.

La huella del escritor incluye `episodes/storage.py`. Reanudar una exportación creada con otra versión del escritor sigue rechazándose por identidad.

## Reproducción y límites

El benchmark reutiliza el recorrido de [episode_pipeline.py](../../benchmarks/episode_pipeline.py). Las carpetas indicadas solo contienen datos sintéticos y pueden reutilizarse para comparar ambas versiones:

```bash
CUDA_VISIBLE_DEVICES='' uv run --extra cuda --extra reinforcement python benchmarks/evaluation_io.py \
  --kind episodes --size representative \
  --work artifacts/benchmarks/evaluation-io/episodes \
  --output artifacts/benchmarks/evaluation-io/episodes.json

CUDA_VISIBLE_DEVICES='' uv run --extra cuda --extra reinforcement python benchmarks/evaluation_io.py \
  --kind evaluation --size representative \
  --work artifacts/benchmarks/evaluation-io/evaluation \
  --output artifacts/benchmarks/evaluation-io/evaluation.json
```

`--size small` ejecuta los tamaños pequeños. `--persist` activa la escritura de predicciones. Para repetir la referencia se usa el mismo script con las versiones de `episodes/storage.py` y `training/reference_run.py` de `4a5bc69`, en una copia separada. El informe registra sus huellas y las de los archivos modificados.

La GPU seguía ocupada por otros procesos. El benchmark adapta explícitamente las entradas y la sincronización a CPU, sin introducir una alternativa CPU en el entrenamiento. Las pruebas completas de entrenamiento CUDA requieren una ventana exclusiva y no forman parte de esta medida. La caché del sistema estaba caliente y otras aplicaciones siguieron activas. No se midieron energía, coste monetario, VRAM ni transferencias CPU/GPU.

La [medición posterior de acumulación por sesión](session-metrics-performance.md) compara la agrupación de errores con sesiones repetidas y con fechas únicas, incluidos lotes pequeños y la escritura de predicciones.
