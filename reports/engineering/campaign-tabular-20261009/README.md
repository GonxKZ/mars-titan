# Ajuste tabular de la campaña A: pasadas, memoria y paridad

Fecha: 9 de octubre de 2026. Tarea [#363](https://github.com/GonxKZ/mars-titan/issues/363), rama `perf/campaign-tabular`.

Este informe mide el recorrido de Ridge y XGBoost en cada ventana de la campaña A, describe los cambios de rendimiento implementados y su paridad, y deja preparadas las medidas que necesitan ajustar modelos. No se ha ejecutado ningún ajuste, paso de optimizador, ronda de boosting ni evaluación científica. Las filas reales solo se han leído para medir el lector, construir matrices cuantizadas y acumular la Gram de Ridge, sin resolver ningún sistema. Las cifras usan GB decimales (10⁹ bytes) salvo cuando se indica GiB.

## Resumen

- Ridge sigue en FP64. Las tres alphas de una ventana comparten las medias, las escalas y la Gram de una sola pasada doble, así que el ajuste de una ventana lee el entrenamiento 5 veces en lugar de 9. La Gram, el lado derecho y las predicciones coinciden bit a bit con la ruta anterior. FP32 y TF32 se midieron y se descartan.
- XGBoost conserva todos sus hiperparámetros. Las configuraciones con el mismo `max_bin` comparten la matriz cuantizada, que se construye 3 o 4 veces por ventana en lugar de 14. La validación se lee una vez por ventana, vive en RAM y en la GPU y ya no se escribe en disco. Cada ronda evalúa solo el árbol nuevo con los mismos bits que la predicción completa, y el resumen del ajuste sale de la matriz sin releer el corpus.
- Las filas leídas por los tabulares de la campaña A bajan de 2,29 a 1,33 miles de millones en Ridge y de 10,84 a entre 2,10 y 2,57 miles de millones en XGBoost. A las 16.000 filas por segundo medidas en CN fold-000 serían de 39,7 a 23,1 horas y de 188,2 a entre 36,4 y 44,7 horas, pero en US+CN fold-012 el lector se midió a 163 filas por segundo, limitado por el disco. Las rondas de boosting no se pueden medir con el bloqueo y no entran en estas cifras.
- En disco solo hay una matriz a la vez, también entre procesos. El pico de páginas es de 25,17 GB (23,44 GiB) en US+CN fold-012 con 256 bins y de 23,1 GB o menos en el resto de casos. La validación de esa ventana (4,2 GB) pasa a RAM. Antes el pico de esa ventana era de 29,4 GB con el formato medido, y el presupuesto de almacenamiento estimaba 31,9 GB con un bit más por valor.
- El objetivo de 25 GB se cumple en GiB, pero las cuatro búsquedas y los dos finalistas con 256 bins de US+CN fold-012 lo superan en 0,17 GB decimales. Con XGBoost 3.3 no hay forma de bajar de ahí sin cambiar `max_bin` o el modelo (ver [disco](#disco-ram-y-vram)).

## Recorrido por ventana

N son las filas de ajuste, V las de validación y H las de validación, calibración y evaluación juntas. Cada ventana tiene 3 ajustes Ridge y 14 de XGBoost (12 configuraciones de búsqueda con la semilla 42 y el caso elegido repetido con 43 y 44).

| Etapa | Antes | Ahora |
| --- | --- | --- |
| Ridge, estadísticas y Gram | 2N por alpha, 6N por ventana | 2N por ventana |
| Ridge, resumen del ajuste y predicciones | N + H por alpha | Igual |
| XGBoost, construcción de la matriz | 2N por configuración, 28N por ventana | 2N por `max_bin` distinto, 6N u 8N por ventana |
| XGBoost, resumen del ajuste | N por configuración | Predicción sobre la matriz, sin lecturas |
| XGBoost, validación | V leída y escrita en npz por configuración | V leída una vez por ventana, en RAM y GPU |
| XGBoost, cada ronda | Lectura de V desde npz y predicción con todos los árboles | Predicción del árbol nuevo sobre la validación residente |
| XGBoost, predicciones reservadas | H por configuración | Igual |

## Cambios implementados

### Ridge

`ridge_statistics` reúne las dos pasadas que antes hacía `fit_ridge_blocks` (medias y varianzas de Chan en la primera, Z'Z y Z'y centrados en la segunda) y `solve_ridge` resuelve una alpha sobre una copia de la Gram. En `tabular_corpus` la clave de las estadísticas es la identidad del manifiesto, la política de entradas, el tamaño de lote y la anchura, que fijan las filas, su orden y el reparto de bloques. La segunda alpha de la misma ventana no lee el entrenamiento para ajustar. El recibo guarda la huella SHA-256 de las estadísticas y si se calcularon o se reutilizaron.

Los bloques llegan en `float32`, que es exacto para estas entradas porque el lector ya entrega `float32` y booleanos. Se copian a dos búferes fijados por turnos con eventos CUDA, de modo que la copia y el GEMM de un bloque se solapan con la lectura del siguiente. La tipificación `(x − media) / escala` se hace en la GPU con la misma resta y división IEEE que hacía NumPy. La predicción tipifica también en la GPU.

La Gram sigue en FP64. Con bloques de 1.024 filas y 1.719 columnas, el GEMM FP64 rinde 0,285 TFLOPS y el FP32 y el TF32 8,1 y 9,2 TFLOPS, pero sus errores máximos relativos al mayor elemento son 8,0·10⁻⁷ y 2,7·10⁻⁵. No dan la misma Gram y se descartan. Además la pasada FP64 acumula unas 46.000 filas por segundo, por encima del lector, y se solapa con él.

La reutilización de la Gram entre ventanas anuales no se ha implementado. Cada ventana tipifica con su propia media y escala, y sumar sobre la Gram anterior cambiaría el orden de las sumas en FP64, así que no daría los mismos bits que la acumulación por ventana del protocolo A.

### XGBoost

`build_external_matrix` construye la `ExtMemQuantileDMatrix` y `fit_external_boosting` acepta una matriz ya construida si su construcción coincide. `SharedWindow` conserva una sola matriz por proceso con la clave de manifiesto, política, lote y todos los parámetros de construcción (`max_bin`, `max_batch_bytes`, presupuestos y ubicación). `max_batch_bytes` forma parte de la clave porque cambia los cortes, como muestra una prueba. La rejilla recorre primero `max_bin`, de modo que las 12 búsquedas construyen 3 matrices. Los finalistas reutilizan la última si comparten `max_bin` y si no construyen una más.

Un cerrojo de archivo junto al directorio compartido (`jobs/.shared-matrix.lock`) lo reserva para el proceso que conserva la matriz. Otro proceso, como una ranura GPU con un proceso por trabajo, espera su turno y atiende una parada mientras espera, sin borrar páginas ajenas ni duplicar el pico. Si el proceso muere, el siguiente borra sus restos antes de construir, y la campaña también los descarta al lanzarse.

`ResidentValidation` sustituye a la caché npz. Lee la validación una vez, comprueba sus valores y copia contiguos a `cuda:0` los primeros bloques hasta 2 GiB, siempre que queden 3 GiB libres para el ajuste. El resto queda en RAM y se copia en cada ronda. La predicción de cada fila no depende del resto del lote, así que la ubicación no cambia ningún valor.

Entre rondas consecutivas del mismo booster solo se evalúa el árbol nuevo. El predictor CUDA de XGBoost suma las hojas de cada fila en orden sobre un acumulador `float32` y después añade la base, así que conservar esa suma y añadirle la hoja nueva da los mismos bits. En cada reinicio se compara toda la validación con la predicción completa y, si difiere, se usa siempre la completa. En cada ronda incremental se vuelve a comparar el primer bloque y el ajuste falla si difiere. El recibo indica en `validation_leaf_sums` qué camino se usó.

El resumen del ajuste (`train_metrics`) se calcula con `booster.predict` sobre la matriz cuantizada, con las claves de fila registradas en la primera pasada de la construcción y la misma acumulación por lotes que `_predict`. Los árboles `hist` dividen en cortes de esa matriz y el predictor de ELLPACK compara el límite inferior de cada bin, así que cada fila sigue las mismas ramas que con sus `float32`.

No cambia ningún hiperparámetro. `nthread` y `cache_host_ratio` se midieron y no alteran cortes ni páginas, pero tampoco aceleran la construcción de forma apreciable y se mantienen como estaban.

### Campaña y almacenamiento

- Antes de cada trabajo, la campaña libera la matriz y la validación si el trabajo no es un ajuste XGBoost, y la Gram si no es un ajuste Ridge. Los traslados también liberan ambas.
- La admisión por disco descuenta las páginas compartidas que el trabajo reutilizará o borrará antes de escribir.
- El ahorro de construcciones exige que los ajustes XGBoost de una ventana se ejecuten seguidos en un mismo proceso, como hace hoy la campaña. Con varias ranuras GPU y un proceso por trabajo, cada proceso construye su propia matriz y espera su turno en el cerrojo, así que el resultado es el mismo pero sin ese ahorro.
- La huella de XGBoost en `campaign_storage` usa el formato medido, `⌈log₂ max_bin⌉` bits por valor y 1.719 columnas, sin caché de validación en disco. `external_cache_plan` usa la misma estimación, y la construcción sigue contrastando los bytes escritos.

## Paridad y pruebas

| Prueba | Qué compara |
| --- | --- |
| `tests/models/test_ridge_statistics_cuda.py` | Gram, Z'y y sumas con bloques `float32` y `float64` frente a la ruta anterior, bit a bit. Estadísticas idénticas, filas cambiadas entre pasadas rechazadas, la Gram compartida intacta tras resolver dos alphas (con `torch.linalg.solve` sustituido) y predicciones tipificadas en la GPU iguales a NumPy, incluidos valores `float64` que no caben en `float32` |
| `tests/models/test_external_matrix_cuda.py` | Dos construcciones con los mismos cortes, páginas y auditoría. Lotes distintos cambian los cortes. Predicción sobre la matriz igual bit a bit a la de los `float32`. Validación residente igual en cualquier ubicación. Rondas incrementales iguales a la predicción completa en cada una de 60 rondas con tres ubicaciones, reinicio con otro booster, respaldo completo si la suma no reproduce y fallo si el estado se altera. Resumen del ajuste desde la matriz igual al de los lotes del lector |
| `tests/training/test_external_shared_cuda.py` | Con el lector real y `xgb.train` sustituido por una excepción: reutilización con el mismo `max_bin`, reconstrucción con otro, lecturas contadas, páginas borradas al liberar o tras un intento interrumpido y claves de fila en el orden del lector |
| `tests/training/test_ridge_shared_cuda.py` | Con `solve_ridge` sustituido: la segunda alpha reutiliza las estadísticas sin leer el ajuste y otro lote las recalcula |
| `tests/training/test_tabular_shared.py` | Claves, liberación sin bloqueo, cerrojo entre procesos, descarte de restos, orden de la rejilla, liberación por tipo de trabajo y admisión por disco, sin CUDA |
| `tests/training/test_campaign_storage.py` y `tests/models/test_external_cache_plan.py` | Huella y plan con el formato medido |

Dos pruebas más comparan modelos ajustados y se omiten mientras rige el bloqueo, antes de abrir fuentes: un ajuste XGBoost con matriz compartida frente a uno con matriz propia, repetido para comprobar el determinismo, y una alpha Ridge con estadísticas compartidas frente a otra con estadísticas propias.

Mutaciones dirigidas de la lógica nueva, cada una aplicada sola y deshecha después: tipificación no IEEE, solución que modifica la Gram compartida, predicción Ridge redondeada a `float32`, claves registradas en cada pasada, clave de matriz sin construcción, estadísticas Ridge compartidas sin la clave del lote, predicción de la matriz sin base, lotes de métricas distintos, validación residente sin el último bloque o sin comprobar su población, temporales interrumpidos sin limpiar, traslados que conservan la matriz, rejilla por profundidad, cierre sin liberar el manejador, hoja de otra ronda, ronda sin comprobar el primer bloque, reinicio sin respaldo completo, base mal leída, ausencia de cerrojo entre procesos, descarte de páginas reservadas, admisión sin descontar la matriz y páginas con un bit de más. Las 22 hacen fallar al menos una prueba. Una mutación más (descartar la propia matriz) resultó equivalente porque el cerrojo ya lo impide, y se eliminó la comprobación redundante.

## Medidas

Equipo: RTX 4070 Laptop (8 GB), Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.0 con CUDA 13.0, XGBoost 3.3.0 y CuPy 14.2.0, perfil de energía sin cambios y otras cargas del equipo activas. Los JSON están en [`measurements.json`](measurements.json).

### Lector y Gram

| Medida | Valor |
| --- | --- |
| Lector y matriz, CN fold-000 | 16.033 filas/s con 60.416 filas y 18.860 filas/s con 102.400 |
| Lector y matriz, US+CN fold-012, primeras 100.352 filas | 163 filas/s, con el proceso esperando al disco y otras cargas activas |
| Gram por bloque de 1.024 filas, ruta anterior (NumPy FP64 y copia paginable) | 23,9 ms, 42.809 filas/s |
| Gram por bloque, ruta actual (fijada y tipificada en la GPU) | 22,2 ms, 46.117 filas/s |
| GEMM FP64 / FP32 / TF32 | 0,285 / 8,1 / 9,2 TFLOPS |
| Error máximo FP32 / TF32 sobre el mayor elemento | 8,0·10⁻⁷ / 2,7·10⁻⁵ |
| 102.400 filas reales con el lector, solo lectura / anterior / actual | 5,43 / 9,21 / 6,16 s, Gram igual bit a bit |
| Predicción Ridge de 40 bloques reales, anterior / actual | 0,40 a 0,94 / 0,08 s en dos medidas, iguales bit a bit |

Con `benchmarks/campaign_tabular.py ridge` y la revisión `7a9e93e3` como referencia, sobre 100.352 filas reales de CN fold-000 (697 columnas constantes): número de filas, medias, escalas, media del objetivo, Gram, Z'y y sumas iguales bit a bit, y las predicciones de un modelo con coeficientes aleatorios también. Con los bloques ya en memoria, la pasada de Gram de referencia tarda 3,64 s y las dos pasadas actuales juntas 4,26 s. La VRAM máxima fue de 119 MiB.

### Matriz cuantizada

Con las 410.342 filas reales de ajuste de CN fold-000 y 1.719 columnas, dos construcciones por `max_bin`:

| `max_bin` | Bytes en disco | Bits por valor | Construcción | VRAM sobre la base | Restos tras cerrar |
| --- | --- | --- | --- | --- | --- |
| 64 | 529.036.152 | 6,0 | 6,9 a 12,5 s | +530 MiB | Ninguno |
| 128 | 617.208.272 | 7,0 | 7,8 a 8,4 s | +530 a +560 MiB | Ninguno |
| 256 | 705.380.392 | 8,0 | 6,9 a 7,1 s | +690 MiB | Ninguno |

Las dos construcciones de cada `max_bin` dan los mismos cortes y páginas. El disco muestreado cada 50 ms durante la construcción nunca supera el tamaño final. Con 200.000 filas aleatorias y 128 bins, `nthread` 1, 4 y 16 y `cache_host_ratio` 0,5 y 1 dan los mismos cortes y páginas (300.826.272 bytes) con tiempos de 1,3 a 1,6 s. `max_batch_bytes` de 16 y 128 MiB cambia los cortes. Con `on_host=True` no se escribe nada en disco.

### Validación por ronda

100.000 filas aleatorias de 1.719 columnas en bloques de 1.024, con bosques escritos a mano de profundidad 6. La ruta anterior lee un npz por bloque (690 MB en total, en la caché de páginas tras la primera lectura) y predice con todos los árboles. La completa predice con todos los árboles desde RAM o desde la copia contigua en la GPU. La incremental recorre las rondas 1 a T con un mismo booster que gana un árbol por ronda y se compara bit a bit con la completa en 8 o 9 rondas. Mediana de tres repeticiones para las rutas completas y de las últimas T/10 rondas para la incremental.

| Árboles | Anterior (npz) | Completa en RAM | Completa en GPU | Incremental en RAM | Incremental en GPU | Iguales |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 0,865 s | 0,330 s | 0,091 s | 0,217 s | 0,018 s | Sí |
| 1.000 | 1,068 s | 0,621 s | 0,242 s | 0,172 s | 0,030 s | Sí |

Las rondas incrementales en la GPU suman 5,0 s para 300 árboles y 20,3 s para 1.000, incluidas las comprobaciones del primer bloque. Con la ruta anterior una sola ronda costaba entre 0,75 s (100 árboles) y 1,07 s (1.000 árboles). En RAM domina la copia de cada ronda y la incremental solo ahorra la evaluación de los árboles.

## Disco, RAM y VRAM

Páginas de la ventana más poblada de cada ámbito con el formato medido:

| Ámbito | Ventana | Filas de ajuste | 64 bins | 128 bins | 256 bins |
| --- | --- | --- | --- | --- | --- |
| US | fold-018 | 12.363.376 | 15,94 GB | 18,60 GB | 21,25 GB |
| CN | fold-012 | 2.275.981 | 2,93 GB | 3,42 GB | 3,91 GB |
| US+CN | fold-012 | 14.639.357 | 18,87 GB | 22,02 GB | 25,17 GB |

En cada momento hay como mucho una matriz en disco. La validación ya no ocupa disco. Antes de este cambio el pico de US+CN fold-012 era de 25,17 GB de páginas más 4,2 GB de validación en npz.

Para bajar de 25 GB decimales en las seis configuraciones de 256 bins de US+CN fold-012 sin cambiar el modelo no hay opción con XGBoost 3.3:

- `cache_host_ratio` solo reparte la caché entre RAM y VRAM cuando `on_host=True`. Con la caché en disco no cambia las páginas (medido). Con `on_host=True` las 25,2 GB irían a RAM y VRAM, y el equipo tiene 32 GB con unos 14 a 18 GB disponibles.
- Menos bits por valor exigen otro `max_bin`, que es un hiperparámetro declarado.
- Quitar columnas constantes o marcar ausencias como NaN cambia la matriz y el formato del modelo.

Las decisiones posibles son aceptar 25,17 GB (23,44 GiB) con la guardia de disco, que admite cada trabajo con el espacio real, o programar los ajustes XGBoost de esa ventana cuando haya más espacio libre.

RAM de un ajuste XGBoost: 6.900 bytes por fila de validación residente (4,2 GB en US+CN fold-012), 17 bytes por fila de ajuste para las claves (0,25 GB) y unos 1,1 GB de construcción medidos. VRAM: de 530 a 690 MiB sobre la base durante la construcción y hasta 2 GiB de validación. La memoria de las rondas no se ha podido medir. XGBoost reserva con CuPy y RMM, fuera del asignador de PyTorch, así que el límite de `set_per_process_memory_fraction` no lo acota.

Declaración por trabajo propuesta para la planificación:

| Trabajo | VRAM | RAM | Disco transitorio |
| --- | --- | --- | --- |
| Ajuste XGBoost | Ranura exclusiva hasta medir las rondas | 6.900·V + 17·N bytes + 1,5 GiB | `⌈N · 1.719 · ⌈log₂ max_bin⌉ / 8⌉` bytes al construir |
| Ajuste Ridge | 512 MiB (pico medido de 226 MiB) | 1,5 GiB | Ninguno |

## Semillas

La búsqueda usa una sola semilla, 42, y el caso elegido de XGBoost se repite con 43 y 44, como declara la campaña. Ridge es determinista: no usa números aleatorios y su resultado solo depende de las filas, su orden y alpha. Por eso la campaña lo ajusta solo con la semilla de búsqueda y no tiene finalistas. Las tres alphas comparten exactamente las mismas estadísticas.

XGBoost declara `subsample` y `colsample_bytree` iguales a 1, así que es probable que la semilla no intervenga y que 43 y 44 repitan los bits de 42. No se ha comprobado porque exige ajustar. Si se confirma, los finalistas aportan una comprobación de determinismo, no variabilidad.

## Proyección de lecturas de la campaña A

Filas leídas con las vistas publicadas de las 45 ventanas, en miles de millones, y horas a 16.000 filas por segundo:

| Ámbito | Ventanas | Ridge antes / ahora | XGBoost antes / ahora | Horas Ridge | Horas XGBoost |
| --- | --- | --- | --- | --- | --- |
| US | 19 | 1,04 / 0,61 | 4,93 / 0,96 a 1,18 | 18,0 / 10,5 | 85,7 / 16,7 a 20,5 |
| CN | 13 | 0,16 / 0,10 | 0,78 / 0,15 a 0,19 | 2,8 / 1,7 | 13,5 / 2,7 a 3,3 |
| US+CN | 13 | 1,08 / 0,63 | 5,13 / 0,98 a 1,21 | 18,8 / 10,9 | 89,1 / 17,1 a 21,0 |
| Total | 45 | 2,29 / 1,33 | 10,84 / 2,10 a 2,57 | 39,7 / 23,1 | 188,2 / 36,4 a 44,7 |

El intervalo de XGBoost depende de si los finalistas reutilizan la última matriz. Lo que queda son sobre todo las construcciones y las predicciones de las particiones reservadas de cada configuración (14H por ventana). Las horas son solo una referencia: el lector rindió 16.000 a 18.900 filas por segundo en CN fold-000 y 163 en el arranque de US+CN fold-012, esperando al disco con otras cargas activas. La tubería de lectura en paralelo de [#363](https://github.com/GonxKZ/mars-titan/issues/363) cambia el caudal, no el número de filas.

El coste por ronda se suma aparte. Antes cada ronda leía la validación desde npz y la predecía con todos los árboles, un coste que crecía con el número de rondas. Ahora crece con las filas de validación y casi nada con los árboles.

## Pendiente tras el desbloqueo

No se ha ejecutado nada de esto. Con el bloqueo levantado:

```bash
# Paridad con modelos ajustados sobre el fixture (matriz y estadísticas compartidas o propias)
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync pytest -q \
  tests/training/test_external_shared_cuda.py tests/training/test_ridge_shared_cuda.py -k after_the_hold

# Una ventana completa con la rejilla declarada: tiempo por ronda, VRAM, RAM y disco reales
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python -m mars_titan.training.tabular_search \
  --config configs/baselines/tabular-historical-masked.json \
  --manifest <vistas>/CN/fold-000/manifest.json \
  --output <salida>/tabular-cn-000
```

En los recibos de cada intento quedan `matrix.constructed`, `validation_cache_bytes`, `validation_device_bytes`, `validation_leaf_sums`, `observed_device_used_bytes_max` y los tiempos. Falta además repetir un caso con las semillas 43 y 44 y comparar las huellas de sus modelos con la de 42.

## Reproducción

Las medidas sin ajuste se repiten con `benchmarks/campaign_tabular.py`, siempre en `cuda:0`:

```bash
uv run --no-sync python benchmarks/campaign_tabular.py gram --blocks 64
uv run --no-sync python benchmarks/campaign_tabular.py ridge --reference 7a9e93e3 --rows 100000 \
  --manifest <vistas>/CN/fold-000/manifest.json
uv run --no-sync python benchmarks/campaign_tabular.py matrix --manifest <vistas>/CN/fold-000/manifest.json \
  --rows 500000 --repeat 2 --cache-root <directorio temporal>
uv run --no-sync python benchmarks/campaign_tabular.py validation --rows 100000 --trees 1000 \
  --cache-root <directorio temporal>
uv run --no-sync python benchmarks/campaign_tabular.py read --manifest <vistas>/US+CN/fold-012/manifest.json
```

La orden `ridge` carga `ridge.py` e `inputs.py` de la revisión indicada como referencia. La orden `validation` usa bosques escritos a mano en el formato JSON de XGBoost, sin `xgb.train`.
