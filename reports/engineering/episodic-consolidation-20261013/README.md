# Consolidación del banco episódico

Recibos del 13 de octubre de 2026 para [#547](https://github.com/GonxKZ/mars-titan/issues/547). La rama parte de `develop` en 7012da85, que ya integra la proyección nativa del codec de [#520](https://github.com/GonxKZ/mars-titan/issues/520) ([#578](https://github.com/GonxKZ/mars-titan/pull/578)).

| Archivo | Contenido |
| --- | --- |
| [`counts.json`](counts.json) | Trabajo contado de cuatro propuestas con el banco lleno e igualdad de bits de todo el flujo, antes y después |
| [`memory.json`](memory.json) | Pico de asignaciones de Python y NumPy de cada propuesta frente a `estimated_peak_bytes` |
| [`mutations.json`](mutations.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`consolidation_counts.py`](consolidation_counts.py), [`consolidation_memory.py`](consolidation_memory.py), [`mutations.py`](mutations.py) | Scripts que generan cada recibo |

Una propuesta del banco es retención de episodios, no ajuste. Ningún recibo construye un modelo ni ejecuta pasos de optimizador, y la protección local de aprendizaje sigue activa.

## Qué se repetía

`RetentionBank.propose` hacía en cada propuesta, sobre todo el banco, dos trabajos que no dependen de los episodios nuevos:

- Convertía desde el enlace la clave de cada registro retenido, 64 números de Python por registro, para formar la matriz de puntos. La clave de un registro no cambia una vez escrito.
- Copiaba el banco con `restore(snapshot())`. Eso pasaba el generador y el recibo por JSON dos veces, volvía a leer los registros retenidos y los metadatos del enlace y validaba con `_check_receipt` un recibo que la propuesta sustituye al terminar.

## Qué cambia

- **`_fork`.** Crea el banco de la propuesta con los bytes de la memoria nativa y el estado del generador, sin el recibo y sin volver a validarlo. El estado es el mismo que da `restore(snapshot())`. `restore` sigue validándolo todo cuando el estado viene de fuera, por ejemplo al reanudar.
- **Claves por ID.** `_keys` guarda la clave FP32 de cada registro retenido como una fila de solo lectura. Cada clave se convierte desde el enlace una vez, al entrar en el banco o en la primera propuesta después de una recuperación. Un banco y sus propuestas comparten las filas, y cada propuesta conserva solo las de los registros que retiene.

No cambian la selección (`select_anchored_medoids`), el formato de los identificadores ni el SHA-256 de la geometría, que forma parte del contrato del recibo.

## Igualdad de bits

- **Pruebas.** `test_proposals_match_the_snapshot_copy_and_full_key_conversion_bit_for_bit` recorre cinco propuestas con las cuatro políticas y compara recibos, instantáneas (identidad, bytes de la memoria, generador y recibo) e IDs retenidos con la ruta anterior, `restore(snapshot())` más la conversión completa de claves.
- **Escala de la campaña.** `consolidation_counts.py` llena un banco de capacidad 1.024 con 12 propuestas de 128 episodios, con `reservoir` y con `anchored` (frontera 8 y 8 candidatos nuevos, como CM-v1), y con los contratos `legacy_v1` y `causal_v2`. En las 48 propuestas coinciden el recibo completo (selecciones, objetivo, contadores y huella de la geometría), la instantánea (bytes de la memoria, generador y recibo) y los registros retenidos (ID, clave, valor, etiqueta y fechas).

La identidad del banco incluye el SHA-256 de `retention_bank.py`, `anchored_medoids.py` y `medoids.py`, así que la comparación fija el mismo valor en los dos árboles. Con `legacy_v1`, esa identidad también siembra el reservorio nativo a través de `scope.representation`. Por tanto, cualquier cambio en esos archivos, aunque sea un comentario, cambia los sorteos de un banco `legacy_v1`. Sin fijar el valor, la propuesta 12 de `legacy_v1` con `reservoir` ya retiene otros registros. Es un comportamiento previo a esta rama. Los lectores de la campaña A usan `causal_v2`, cuyo reservorio no depende del ámbito, y en ellos el hash solo cambia la identidad y los bytes de la instantánea.

## Trabajo contado

Cuatro propuestas con el banco lleno (1.024 registros retenidos y 128 nuevos, 1.152 clientes), `develop` frente a la rama:

| Recuento en las cuatro propuestas | Antes | Después |
| --- | --- | --- |
| Lecturas de claves de registros ya retenidos | 4.096 | 0 |
| Llamadas a `json.dumps` | 32 | 4 |
| Caracteres de `json.dumps` | 115.826 a 119.448 | 4.404 a 4.848 |
| Llamadas a `json.loads` | 20 | 0 |
| Validaciones del recibo copiado (`_check_receipt_selection`, que solo llama `_check_receipt`) | 4 | 0 |
| Llamadas a SHA-256 | 8 (`reservoir`) y 520 (`anchored`) | iguales |

Cada propuesta deja de crear unos 65.536 números de Python para las claves. Las cuatro serializaciones que quedan son la identidad del banco nuevo. Según el código, cada propuesta deja de pedir al enlace una segunda lectura de los registros retenidos y `snapshot_metadata`, que solo servían para validar el recibo copiado. La copia de la memoria nativa (`snapshot_bytes` y `restore_bytes`) se mantiene. El SHA-256 no cambia: son la identidad, la huella de la geometría y, con `anchored`, el orden de los candidatos nuevos de `_centers`.

## Memoria

Con tracemalloc, el pico de asignaciones de Python y NumPy de cada propuesta con el banco lleno está entre 4,85 y 4,87 MB con `reservoir` y entre 5,09 y 5,10 MB con `anchored`. Los valores declarados en `estimated_peak_bytes` son 20,4 y 20,5 MB y no cambian. La rama sube el pico entre 1 y 10 KB, un 0,2 % como mucho. El pico llega durante la selección, cuando la lista de claves de Python ya se ha liberado, y la propuesta añade el diccionario de claves del banco nuevo y las filas de los episodios nuevos. Las filas de un banco lleno ocupan unos 0,4 MB que se conservan entre propuestas. Caben de sobra en los 4.096 bytes por registro que reserva el presupuesto propio de la propuesta. tracemalloc no ve las asignaciones del enlace C++, así que el pico medido es una cota inferior del total.

## Mutaciones

`mutations.py` cambia un único fragmento de `retention_bank.py`, ejecuta `tests/memory/test_retention_bank.py` con `-x` y sin orden aleatorio y restaura el archivo. Las cinco hacen fallar una prueba.

| Mutación | Qué rompe | Prueba que falla |
| --- | --- | --- |
| `fork_without_generator` | la propuesta no hereda el generador | `test_proposals_match_the_snapshot_copy_and_full_key_conversion_bit_for_bit[uniform]` |
| `fork_without_memory` | la propuesta no hereda la memoria | `test_proposals_keep_original_records_and_leave_previous_snapshot_unchanged[reservoir]` |
| `proposal_without_key_cache` | la propuesta no guarda las claves | `test_proposals_convert_each_retained_key_once[reservoir]` |
| `cache_keeps_every_client` | se guardan también las claves descartadas | la misma |
| `writable_cached_rows` | las filas compartidas se pueden escribir | la misma |

## Medidas de tiempo

Las rondas se tomaron el 14 de octubre de 2026 con los mismos eventos, opciones y condiciones que las de [#549](../titans-event-path-20261013/README.md#medidas-de-tiempo): vista `US+CN/fold-012` de las vistas v3.1 de A v2, 88.028 filas y 11 llamadas a `_NoStepOptimizer` sin ningún paso, FP32 estricto, dos hilos, perfil de energía de ahorro y un trabajo `memslot gpu` por ronda. El recibo es [`abba-20261014.json`](abba-20261014.json). Los árboles son A = bb19f7be (`develop` con #591) y B = 89913902, el commit de [#593](https://github.com/GonxKZ/mars-titan/pull/593), sin más cambios entre ellos.

| Modelo | Filas/s antes | Filas/s después | Cociente de medias | Peor par |
| --- | --- | --- | --- | --- |
| `mars_titan_m1` | 6.878 y 6.972 | 6.671 y 7.078 | 0,99 | 0,96 |
| `cm_v1_bm` (B+M) | 5.822 y 5.708 | 6.026 y 6.039 | 1,05 | 1,04 |
| `cm_v1_bcm` (B+C+M) | 5.908 y 5.683 | 6.026 y 5.661 | 1,01 | 0,96 |

El peor par divide la ronda más lenta de después entre la más rápida de antes. La memoria no cambia en ningún modelo (402 MiB de pico asignado y 492 o 494 MiB reservados). Las capturas de las cuatro rondas de cada modelo coinciden bit a bit entre árboles y dentro de cada árbol: nombres, gradientes de las 11 llamadas (77 tensores) y las 3.347 predicciones pendientes al terminar.

Solo B+M mejora de forma clara: sus dos rondas de después superan a las dos de antes. En M1 y B+C+M las rondas de los dos árboles se solapan, así que el cambio queda dentro de la variación entre rondas. Las cuentas de [Trabajo contado](#trabajo-contado) siguen siendo ciertas, pero en este tramo las propuestas del banco no pesan lo bastante para que retirar ese trabajo se note en el caudal de M1 o de B+C+M.

## Pendiente

- **Admisión por partes.** El perfil con py-spy de M1, B+M y B+C+M sobre `develop` debe decir qué parte del recorrido ocupan `propose@memory/retention_bank.py` y `select_anchored_medoids@cm/anchored_medoids.py` después de esta rama.
- **Núcleos de los medoids.** La revisión de `_bounded_minimum`, `_compute`, `_sum` y `_reference_chain` en `cm/medoids.py` y `cm/anchored_medoids.py` depende de ese perfil, porque la issue la condiciona a lo que domine tras retirar este trabajo.

Órdenes para repetir las medidas desde la raíz de cada árbol, con `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src`, el enlace en `MARS_TITAN_EPISODIC_NATIVE`, los eventos de una vista v3.1 (`benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`) y un trabajo `memslot gpu` por ronda, alternando los árboles:

- `python benchmarks/chronological_replay.py VISTA/manifest.json TRABAJO --family readout --arm mars_titan_m1 --option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8 --events eventos.pkl --capture CAPTURA.pt --output resultados.jsonl --label ETIQUETA`. Para B+M y B+C+M se usa `--family cm_readout --arm cm_v1_bm` y `--arm cm_v1_bcm` con las mismas opciones.
- `python reports/engineering/profiling-20261010/compare_capture.py DIRECTORIO salida.json antes:despues` para los bits de las rondas.
- `py-spy record --rate 50 --format raw` y `MARS_TITAN_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` sobre la misma orden, resumidos con `benchmarks/collapsed_profile_summary.py` (raíz `_train_pass@training/mars_titan_run.py` y funciones como `propose@memory/retention_bank.py` y `select_anchored_medoids@cm/anchored_medoids.py`) y `benchmarks/nsys_range_summary.py`.
- `python benchmarks/anchored_medoid_options.py salida.json` para los núcleos de los medoids.

## Reproducción

- Cuentas e igualdad de bits: `python consolidation_counts.py ÁRBOL salida.pkl` en cada árbol, con la misma semilla, y comparación de `states` entre los dos archivos.
- Memoria: `python consolidation_memory.py ÁRBOL salida.json`.
- Mutaciones: `python mutations.py ÁRBOL salida.json`.
- Pruebas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/memory tests/cm tests/posttraining/test_readout_adapters.py tests/training/test_candidate_run.py tests/training/test_cm_v1_factorial.py tests/training/test_mars_titan_run.py tests/test_learning_hold_lowering.py tests/training/test_learning_hold_inventory.py`. Con `cuda:0` visible y el enlace compilado desde 7012da85 pasaron 962 pruebas en 34 min 48 s.
