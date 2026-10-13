# Perfilado de CPU, memoria y GPU (10 de octubre de 2026)

Este informe responde a #488. El [informe de núcleos de la campaña](../campaign-kernels-20261009/README.md) dejó dos huecos. En mac_online con bloque 1.024 la GPU trabajaba en torno al 19 % del tiempo y el 48 % de la CPU quedaba fuera de los operadores de PyTorch, sin saber en qué. En RAM, las declaraciones de memoria por trabajo salían de picos de RSS sin atribuir a ninguna asignación. Aquí se reparte el tiempo de Python por función con py-spy, se sigue cada fase en la GPU con rangos NVTX y Nsight Systems y se atribuye el pico de RAM por asignación con memray.

Ninguna medida aplica pasos de optimizador ni entrena. Los recorridos de ajuste usan el optimizador sustituto de `benchmarks/chronological_replay.py`, que solo cuenta llamadas y guarda gradientes, y cada medida comprueba al final que ningún parámetro ha cambiado. No se guarda ningún checkpoint ni se evalúa ninguna predicción. Las cifras describen el coste de cálculo y no dicen nada de la calidad predictiva.

## Entorno

| Elemento | Valor |
|---|---|
| CPU | AMD Ryzen 9 8945HS, 8 núcleos y 16 hilos, un nodo NUMA |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, controlador 595.91.07 |
| Memoria | 32 GB, compartida con otras tareas de la sesión |
| Software | Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0 |
| Herramientas | py-spy 0.4.2 (`uv tool`), memray 1.20.0 (grupo `dev`), Nsight Systems 2026.3.2 |
| Energía | Perfil `power-saver` sin cambios. No se tocaron relojes, ventiladores ni límites |
| Seguridad | `perf_event_paranoid` 4, `RmProfilingAdminOnly=1` y `ptrace_scope` 1 sin cambios |
| Precisión | FP32 estricto, sin TF32 en cuBLAS ni en cuDNN y `float32_matmul_precision="highest"` |

Cada recorrido con GPU se ejecutó en la plaza exclusiva `memslot gpu` y los de solo CPU en `memslot suite`, con dos hilos de BLAS y OpenMP. La máquina estaba compartida con suites y compilaciones de otras ramas, con una carga media entre 6 y 33. Dos ejecuciones de develop sin herramienta difieren un 22 % entre tandas en mac_online y un 48 % en M1, así que las comparaciones se hacen dentro de la misma tanda y una diferencia menor que esa dispersión no se distingue del ruido. El guardián térmico no congeló el cálculo durante ninguna de estas medidas.

## Recorridos medidos

Los recorridos de tiempo repiten con `--events` los 254 eventos guardados de US+CN `fold-012` de la edición v3 con máscaras, desde el evento 127, con dos actualizaciones de calentamiento y ocho tramos medidos (151.797 filas y 11 llamadas al optimizador sustituto). `replay_stored_events.py` es el envoltorio de lectura bloqueada de la medida de medoids de `perf/campaign-kernels-wiring`. Reutiliza los índices guardados y falla si el recorrido intenta leer muestras de la vista, así que la medida aísla el cálculo. Los dos brazos son:

- Titans-MAC `titans_mac_online` con bloque y acumulación de 1.024 filas.
- Lector MARS-TITAN M1 con bloque de 512 filas.

El código es develop en b92e07a0 y esta rama sobre la misma base. Dos tandas ejecutan develop, la rama con los rangos apagados, py-spy con todas las muestras, py-spy con `--gil` y los rangos encendidos sin herramienta, y la primera además Nsight Systems. Otras dos tandas repiten py-spy con `--nonblocking` junto a la rama sin herramienta. Las cifras por ejecución están en `evidence/replay-runs.json`.

## Sobrecarga del muestreo

py-spy detiene por defecto el proceso en cada muestra para leer una pila coherente. Con `--nonblocking` lee la memoria del proceso sin detenerlo y a cambio pierde algunas muestras cuya pila cambia durante la lectura.

| Brazo | Modo | Tanda 1 (s) | Tanda 2 (s) | Frente a la rama sin herramienta | Muestras perdidas |
|---|---|---:|---:|---:|---:|
| mac_online | Rama sin herramienta | 10,46 | 13,51 | | |
| mac_online | py-spy bloqueante | 13,95 | 16,47 | +33 % y +22 % | 0 |
| mac_online | py-spy bloqueante con `--gil` | 17,06 | 13,57 | +63 % y +0,5 % | 0 |
| mac_online | Rama sin herramienta (sin pausa) | 15,15 | 13,10 | | |
| mac_online | py-spy `--nonblocking` | 12,35 | 11,71 | −18 % y −11 % | 9,0 % y 9,6 % |
| mac_online | py-spy `--nonblocking --gil` | 14,54 | 13,87 | −4 % y +6 % | 13,6 % y 13,2 % |
| M1 | Rama sin herramienta | 157,5 | 175,6 | | |
| M1 | py-spy bloqueante | 213,8 | 333,5 | +36 % y +90 % | 0 |
| M1 | py-spy bloqueante con `--gil` | 161,2 | 243,7 | +2 % y +39 % | 0 |
| M1 | Rama sin herramienta (sin pausa) | 121,7 | 107,2 | | |
| M1 | py-spy `--nonblocking` | 120,5 | 109,9 | −1 % y +3 % | 1,1 % y 1,2 % |
| M1 | py-spy `--nonblocking --gil` | 110,8 | 104,8 | −9 % y −2 % | 7,9 % y 9,4 % |

El modo bloqueante alarga el recorrido entre un 22 % y un 90 %. El modo sin pausa no tiene un coste distinguible del ruido, pero pierde entre el 1 % y el 14 % de las muestras. Los dos modos ordenan igual las funciones principales y sus fracciones difieren hasta 7 puntos en mac_online y 10 en M1 (tablas siguientes), así que las conclusiones no dependen del modo. Para medir conviene `--nonblocking` y comprobar el número de muestras perdidas que imprime py-spy al terminar.

`--native` añade los marcos de C++ pero no sirve aquí. Desenrollar las pilas de libtorch de todos los hilos tarda más que el intervalo de muestreo y mac_online pasó de 37 s a más de cinco minutos con la GPU trabajando mientras el proceso estaba detenido. Esa captura se descartó.

## Tiempo de Python en mac_online

`pyspy_split.py` toma el hilo principal y las muestras dentro de `_train_pass`. Compara la tasa de muestras de la captura completa con la de `--gil`, que solo guarda las muestras en las que el hilo tiene el GIL. PyTorch suelta el GIL mientras ejecuta un operador llamado desde Python y `hashlib` también lo suelta con búferes grandes, así que la fracción con el GIL es el tiempo del intérprete y el resto es el de dentro del código nativo llamado desde esa función. Como las capturas son independientes, el cociente puede pasar de 1 en funciones que solo usan el intérprete.

El hilo principal tiene el GIL durante el 31 % del recorrido sin pausa (sin corregir las muestras perdidas) y el 37 % del bloqueante. Esa es la parte del intérprete. El resto del tiempo activo se va en código nativo llamado desde Python.

| Función (inclusiva) | Sin pausa | Bloqueante | Fracción con GIL | Qué hace |
|---|---:|---:|---:|---|
| `financial_run._observe` | 54,2 % | 52,2 % | 0,40 | Emisión de las predicciones del instante |
| `financial_run._replay` | 41,6 % | 43,9 % | 0,44 | Repetición de los bloques maduros y backward |
| `financial.prepare` | 40,0 % | 46,6 % | 0,37 | Forward de Titans-MAC en la emisión y en la repetición |
| `financial_inputs.validated_cpu_batch` | 15,9 % | 15,0 % | 0,42 | Validación y copia de las entradas de un bloque en CPU |
| `financial_inputs._digest` | 14,5 % | 12,2 % | 0,22 | Huella SHA-256 de las entradas de un bloque |
| `state.check_finite` | 13,8 % | 11,7 % | 0,12 | Comprobación de finitud de un tensor |
| `financial_inputs.from_validated` | 10,8 % | 9,0 % | 0,18 | Verificación y copia al dispositivo |
| `neural_memory.update` | 9,6 % | 9,2 % | 0,20 | Actualización de la memoria neuronal |
| `financial._validate_state` | 6,6 % | 8,9 % | 0,80 | Comprobación del estado de los flujos |
| `config.canonical` y `json.dumps` | 3,7 % | 5,7 % | ≥ 0,9 | JSON canónico de huellas y configuraciones |
| `financial_inputs.device_tensor` | 4,2 % | 3,5 % | 0,14 | Copia a memoria fijada y al dispositivo |

Con esas pilas el tiempo de CPU fuera de los operadores tiene cuatro destinos concretos.

1. La huella de las entradas. `_digest` ocupa entre el 12 % y el 15 % del recorrido. Se calcula dos veces por bloque: al crear el lote validado en `validated_cpu_batch` (7,8 % del recorrido sin pausa) y otra vez en `CPUDecisionBatch.verify`, que `from_validated` llama justo después dentro del mismo `_observe` (6,5 %). Las dos recorren los mismos búferes de solo lectura.
2. Las comprobaciones de finitud. `check_finite` lanza `isfinite` y `all` por cada tensor comprobado y deja la lectura para el final de la preparación. Sus llamadores principales son `DecisionBatch.verify` sobre cada entrada (4,8 %), `neural_memory.validate_state` (2,7 %) y `neural_memory.update` (2,4 %). La espera agrupada aparece aparte en `state.resolve` (3,7 %).
3. Las comprobaciones de estado y el JSON canónico. `_validate_state` y las huellas de configuración serializan a JSON en cada instante (`_state_size`, `_config_id`, `_usage` y la identidad de cada lote). Es tiempo de intérprete puro.
4. El bucle de repetición en Python. `_replay` tiene un 8,3 % de tiempo propio con fracción de GIL 0,62.

Nsight Systems con los rangos encendidos da el reparto por fase de un recorrido trazado de 27,4 s, más largo que sin traza porque la herramienta alarga el lado de CPU.

| Rango | Tiempo | Veces |
|---|---:|---:|
| `titans.update` | 14,95 s | 11 |
| `titans.backward` (dentro de `update`) | 14,79 s | 11 |
| `titans.observe` | 11,91 s | 88 |
| `titans.prepare` (en la emisión y en la repetición) | 4,45 s | 226 |
| `titans.validate` | 2,52 s | 88 |
| `titans.emit` | 0,53 s | 88 |
| `titans.labels` | 0,30 s | 89 |
| `titans.clip` | 0,08 s | 11 |
| `titans.optimizer` | 0,03 s | 11 |
| `titans.truncate` | 0,02 s | 11 |
| `titans.read` | 0,01 s | 89 |

Los núcleos suman 6,12 s y las copias y rellenos 0,26 s, así que la GPU trabaja en torno al 23 % del recorrido trazado. El recorrido lanza 456.611 núcleos, con una media de 13,4 µs por núcleo y 442.784 llamadas a `cudaLaunchKernel`. La atención eficiente en memoria de cutlass en FP32 suma el 22,5 % del tiempo de GPU y los rellenos de `fill_` el 5,8 %. Estas cifras coinciden con el informe de núcleos. Lo que limita es el número de operaciones pequeñas lanzadas desde Python, no un núcleo concreto. La medida de CUDA Graphs en Titans-MAC pertenece a la rama de núcleos y no se repite aquí.

## Tiempo de Python en el lector M1

En M1 el hilo principal está activo el 99 % del recorrido pero solo tiene el GIL entre el 8 % y el 12 %. Casi todo el tiempo está en NumPy.

| Función (inclusiva) | Sin pausa | Bloqueante | Fracción con GIL |
|---|---:|---:|---:|
| `mars_titan_run._admit` | 79,4 % | 72,2 % | 0,05 |
| `anchored_medoids._background` | 76,2 % | 66,2 % | 0,01 |
| `medoids._distance` | 76,1 % | 66,2 % | 0,01 |
| `mars_titan_run._observe` | 19,4 % | 26,7 % | 0,17 |
| `episodic_codec.encode` | 12,5 % | 11,0 % | 0,03 |
| `numpy.einsum` (dentro de `encode`) | 10,9 % | 9,8 % | 0,01 |

Nsight Systems da 175,9 s de `readout.admit` sobre 236,8 s de recorrido trazado (74 %), 55,1 s de `readout.observe` (con 23,9 s de `readout.encode`, 10,2 s de `readout.prepare`, 5,0 s de `readout.snapshot` y 3,5 s de `readout.apply`) y 4,6 s de `readout.update`. La GPU trabaja el 1,3 % del recorrido (2,90 s de núcleos y 0,24 s de copias).

Ese coste es el que ataca ce9542e8 de `perf/campaign-kernels-wiring` (10 de octubre de 2026, sin integrar al escribir este informe), que reutiliza las distancias variables y acota el fondo fijo de los medoids con los mismos bits. Con ese commit, el mismo recorrido de M1 bajo py-spy bloqueante a 50 Hz alcanza 3.892 filas/s, frente a entre 804 y 1.416 filas/s de develop y de la rama sin herramienta en estas tandas. Las cifras no están emparejadas y la primera incluye la sobrecarga de py-spy. La admisión baja al 13,7 % de las muestras y `episodic_codec.encode` pasa a ser el primer coste con el 30 %. Por eso este informe no repite la medida de los medoids.

## Memoria

Las medidas de memoria usan una vista de `fold-012` de la v3 con enlaces duros a las muestras que seguían presentes y verificadas, porque la codificación de la v3.1 estaba sustituyendo las de la v3. La vista conserva 4.130 de 5.008 activos (3.424 de US y 706 de CN), entre el 82,4 % y el 82,7 % de las filas de cada partición, y recalcula sus recuentos (`evidence/memory-view.json`). Los picos que dependen del número de activos serán algo mayores en la vista completa y deben repetirse sobre la v3.1.

Cada recorrido se mide sin memray, con el asignador por defecto de Arrow (mimalloc), para obtener el pico real del proceso, y con `memray run --native --aggregate` y `ARROW_DEFAULT_MEMORY_POOL=system` para atribuirlo. memray alarga el recorrido entre 2,6 y 5,9 veces y sube el RSS hasta 1 GiB. El RSS se lee con `ru_maxrss` y con un muestreo cada 0,1 s que marca la fase del pico. El pico del grupo de control de `memslot` incluye la caché de páginas y no sirve para comparar.

| Recorrido | Pico RSS sin memray | Fase del pico | Memoria viva en el pico según memray | Declaración |
|---|---:|---|---:|---:|
| Índice de ajuste (9.356 eventos) y validación (747) | 3,27 GiB | Índice de ajuste | 2,99 GB | 9.216 MiB por trabajo cronológico en US+CN |
| Lectura de los 100 últimos instantes con caché de 4.096 MiB | 5,41 GiB | Lectura | 4,85 GB | 9.216 MiB por trabajo cronológico en US+CN |
| Validación residente de XGBoost sin rondas | 5,07 y 5,23 GiB | Copia a la GPU | 11,7 GB, con proyecciones de CUDA (ver abajo) | 3.465.914.636 bytes residentes y 8.192 MiB por proceso |
| Matriz de ajuste de XGBoost sin rondas, de 110.095 a 739.214 filas | De 1,53 a 1,74 GiB | Construcción | 0,52 GB del montículo y 21,5 GB de proyecciones | 8.192 MiB de RAM y 7.000 MiB de VRAM por proceso |

### Índice del lector cronológico

El 43 % de la memoria viva en el pico (1,30 GB) sale de la ordenación con DuckDB de `corpus_source._ordered_parquet`. `duckdb_native_split.py` la reparte por su pila nativa: 0,61 GB son metadatos Parquet leídos por `FileMetaData::read`, 0,51 GB el resto del lector Parquet y 0,17 GB el gestor de bloques de DuckDB. `memory_limit` (256 MiB) solo acota el gestor de bloques, como ya recogía la [memoria de la ordenación del corpus](../../../docs/engineering/corpus-sort-memory.md), así que el 87 % de esa memoria queda fuera del límite declarado. Las etiquetas (`corpus_inputs._labels`) suman otro 35 % (1,06 GB entre sus tres líneas).

### Lectura de los últimos instantes

La caché de grupos decodificados (`corpus_inputs._vectors`) ocupa 2,93 GB, el 60 % de la memoria viva en el pico, y el lector informa de un máximo de 2.921.774.322 bytes en caché, por debajo de los 4.096 MiB configurados. Le siguen las ventanas de precios (`_prices`, 0,61 GB), las etiquetas de la partición (0,38 GB) y las etiquetas por lote (0,21 GB). Con el asignador del sistema en lugar de mimalloc el pico apenas cambia (5,48 frente a 5,41 GiB).

La declaración de 9.216 MiB de las familias cronológicas parte de un pico de 6,6 GiB medido en los últimos instantes de esta misma ventana con la vista completa, más el contexto CUDA y el entrenador. El índice y la lectura se miden aquí en procesos separados y los dos quedan por debajo. Con el 82,7 % de los activos no se puede afirmar más que eso.

### Validación residente de XGBoost

La validación tiene 502.015 filas y 1.719 columnas en 491 bloques. Lo residente ocupa 3.463.903.500 bytes, exactamente 6.900 bytes por fila: 6.876 de valores float32 y 8 de cada clave (objetivo float64, mercado `<U2` y fecha de predicción `datetime64[us]`). La declaración que `run_external_reference` pasa al plan de caché es 3.465.914.636 bytes, la misma cifra más 4.096 bytes por bloque, así que acierta con 2.011.136 bytes de margen. memray atribuye 3,45 GB a `tabular_corpus._matrix`, que es justamente la matriz residente.

El pico del proceso (5,07 y 5,23 GiB en las dos ejecuciones sin memray) suma unos 0,9 GiB al arrancar (importaciones y contexto CUDA), los 3,23 GiB residentes y alrededor de 1 GiB transitorio de la lectura (etiquetas y precios). Al liberar la validación el proceso vuelve a unos 1,9 GiB. La copia a la GPU ocupa 2.140.471.296 bytes, que es lo que cabe en el presupuesto de dispositivo. La declaración de 8.192 MiB de XGBoost cubre este pico, pero este recorrido no construye la matriz de ajuste ni hace rondas, así que no la valida completa. La matriz se mide en el apartado siguiente. La primera ejecución sin memray contó los bloques después de liberarlos. Sus medidas de memoria son válidas, pero el recuento se repitió con el arnés corregido (`validation-plain-r2`).

En los procesos con CUDA el máximo de memray no es memoria del anfitrión. Para la validación marca 11,7 GB, pero 4,77 GB son proyecciones `mmap` creadas al iniciar CUDA en `external_corpus._device_budget` y 2,18 GB corresponden a `cupy.empty`, la reserva en la GPU que el controlador proyecta en el espacio de direcciones del proceso. Su RSS con memray fue de 5,44 GiB. En esos procesos hay que comparar el RSS y usar memray solo para atribuir las asignaciones del anfitrión.

### Matriz de ajuste de XGBoost

`memory_profiles.py matrix` construye la matriz cuantizada de ajuste igual que `external_corpus._execute`. Usa la misma factoría de lotes con las claves de cada fila (`_TrainRows`), la construcción de `tabular-historical-masked.json` y `build_external_matrix`, que recorre el iterador dos veces y escribe las páginas ELLPACK en disco. No ajusta ninguna ronda. Con `--validation` lee después la validación residente y la copia a `cuda:0` con la matriz todavía viva, en el mismo orden que el ajuste. Registra el RSS cada 0,1 s con su parte anónima (`RssAnon`), la memoria del proceso en la GPU según `nvidia-smi` cada 0,5 s y, cada segundo, la caché del lector y la memoria reservada por Arrow.

La matriz de la ventana completa escribiría entre 18,9 y 25,2 GB de páginas, así que se midió sobre tres vistas de `fold-012` (v3) con uno de cada 100, 33 y 16 activos cuyas tres tablas conservan su huella (`view_subset.py`, con enlaces duros y recuentos recalculados en `evidence/xgboost-matrix-views.json`). Tienen 37, 123 y 244 activos y 110.095, 364.104 y 739.214 filas de ajuste. Con 64 bins hay cuatro ejecuciones por tamaño (cinco en el intermedio) y con 256 bins otras ocho en los dos primeros, más una con memray (`evidence/xgboost-matrix-runs.json`). La carga media estuvo entre 18 y 27 y el guardián térmico no congeló el cálculo.

| Bins | Filas de ajuste | Pico RSS en la construcción | Parte anónima | Caché del lector | Claves de las filas | VRAM del proceso | Páginas en disco |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 110.095 | 1,588 a 1,617 GiB | 1,105 a 1,110 GiB | 16,2 MB | 1,9 MB | 658 MiB | 141.940.736 bytes |
| 64 | 364.104 | 1,601 a 1,656 GiB | 1,118 a 1,168 GiB | 54,2 MB | 6,2 MB | 658 MiB | 469.423.488 bytes |
| 64 | 739.214 | 1,711 a 1,741 GiB | 1,260 a 1,290 GiB | 110,0 MB | 12,6 MB | 658 MiB | 953.036.472 bytes |
| 256 | 110.095 | 1,532 a 1,566 GiB | Sin medir | 16,2 MB | 1,9 MB | 786 MiB | 189.254.000 bytes |
| 256 | 364.104 | 1,604 a 1,668 GiB | Sin medir | 54,2 MB | 6,2 MB | 818 MiB | 625.896.984 bytes |

El pico llega siempre durante la construcción y el proceso parte de 0,81 a 0,98 GiB (importaciones y contexto CUDA). La parte del RSS respaldada por archivos (bibliotecas y CUDA) se queda entre 0,35 y 0,48 GiB en todos los tamaños, así que lo que crece es memoria anónima. memray atribuye a la ejecución de 364.104 filas con 256 bins 0,52 GB de montículo vivo en el pico (familia `malloc`) y 21,5 GB de proyecciones `mmap` de CUDA y de la caché de páginas, que no son RSS del anfitrión (`evidence/memray-matrix.json`). Con 256 bins el RSS queda en el mismo intervalo que con 64 o algo por debajo.

Las páginas coinciden con la estimación densa del plan (`external_cache_plan`, 6 bits por valor con 64 bins y 8 con 256) con entre 695 y 4.822 bytes más por matriz. La estimación no es una cota estricta, pero el desfase es de kilobytes y la construcción vuelve a comprobar los bytes escritos contra `max_disk_cache_bytes` en cada lote. Para la ventana completa supone 18,9 GB con 64 bins y 25,2 GB con 256, por debajo de los 32 GiB declarados. Durante estas medidas el disco tenía unos 31 GB libres, de modo que la construcción con 256 bins cabe con poco margen y depende de lo que ocupen otras tareas.

#### Estimación para la ventana completa

De lo que crece con el tamaño, la caché del lector (unos 450 KB por activo, porque guarda etiquetas y precios de cada activo para todas las particiones) y las claves de las filas (17 bytes por fila) se miden directamente. Tras restarlas, un ajuste lineal sobre las doce ejecuciones de 64 bins con caché registrada da 1,569 GiB más 50 bytes por fila, con un error típico de 24 y un intervalo del 95 % entre −3 y 104 bytes por fila (`matrix_scaling.py`, `evidence/xgboost-matrix-scaling.json`). La ventana completa tiene 14.639.357 filas de ajuste y 609.090 de validación. La caché del lector queda en su límite de 1 GiB, porque con unos 5.000 activos ocuparía más de 2 GB, y las claves suman 0,23 GiB.

La validación residente se suma en el orden de `_execute`, con 6.900 bytes por fila (3,91 GiB). En las dos ejecuciones de validación sola, lo que el pico no explica con lo residente ni con lo que queda al liberarla no pasa de 17 MB. Lo que queda (1,0 y 1,1 GiB sobre el arranque) coincide con la caché del lector en su límite, porque los 4.130 activos de aquella vista la llenan, y ya está contado en la matriz. La suma usa el pico de la construcción y no lo retenido después, así que es conservadora en unos 0,2 GiB.

| Pendiente | Matriz sola | Matriz y validación residente | Sobre el arranque del proceso |
|---|---:|---:|---:|
| Extremo inferior (−3 bytes por fila) | 2,76 GiB | 6,69 GiB | 5,78 GiB |
| Central (50 bytes por fila) | 3,49 GiB | 7,42 GiB | 6,51 GiB |
| Extremo superior (104 bytes por fila) | 4,22 GiB | 8,15 GiB | 7,24 GiB |

La matriz sola cabe con holgura en los 8.192 MiB (8,00 GiB) declarados para XGBoost. La secuencia que ejecuta el ajuste, con la validación residente leída sobre la matriz viva, queda entre 6,7 y 8,2 GiB por proceso, así que la declaración cubre el valor central con 0,6 GiB de margen y el extremo superior la supera. Si la declaración se interpreta como lo que añade el trabajo al proceso de la campaña, que ya tiene el contexto CUDA, el extremo superior queda en 7,24 GiB. Ninguna de las dos cifras incluye las rondas, que leen las páginas de disco y añaden sus propios búferes, y que no se pueden medir mientras siga el bloqueo de aprendizaje.

En la GPU, la construcción ocupa 658 MiB con 64 bins en los tres tamaños, y 786 y 818 MiB con 256 bins, contexto CUDA incluido. Aunque esos 32 MiB crecieran en proporción a las filas, la ventana completa no pasaría de unos 2,6 GiB. A eso se añade durante el ajuste la validación copiada a la GPU, hasta 2 GiB (`VALIDATION_DEVICE_BYTES`), lejos de los 7.000 MiB declarados. La construcción tardó entre 84 y 88 s por millón de filas con la máquina cargada, entre 20 y 22 minutos para la ventana completa.

Conclusión para el plan: la estimación del disco es exacta y la VRAM de la construcción queda muy por debajo de su declaración, pero la RAM de la secuencia completa no queda validada. La extrapolación multiplica por 20 el mayor tamaño medido. Antes de entrenar hay que medir la secuencia de matriz y validación sobre la v3.1 completa y, si supera los 8.192 MiB, subir la declaración o reducir una parte medida, como la caché del lector durante las pasadas de XGBoost.

## Rangos NVTX

`src/mars_titan/nvtx_ranges.py` abre rangos por fase en `financial_run.py`, `mars_titan_run.py` y `financial_observations.py` (la lista completa está en [Perfilado](../../../docs/engineering/profiling.md)). Están apagados por defecto y se encienden con `MARS_TITAN_NVTX=1` al importar el módulo.

| Comprobación | Resultado |
|---|---|
| Datos reales en `cuda:0` (`compare_capture.py`) | develop, la rama apagada y la rama encendida dan los mismos 462 gradientes en 11 pasos y las mismas 1.265 predicciones pendientes, bit a bit. La rama encendida con `replay_stored_events.py` coincide también con develop con el envoltorio original (`evidence/nvtx-parity.json`) |
| `tests/training/test_nvtx_parity.py` | Titans-MAC sin control y con penalización, con y sin acumulación, en CPU y en `cuda:0`, y el lector con admisiones M0, M1 y M3: mismos gradientes, historial, selección y auditoría, y parámetros sin cambios |
| `tests/tooling/test_nvtx_ranges.py` | Cierre ante excepciones sin tragarlas, cierre del lector interno, rangos alrededor de cada `next` y no del consumidor, equilibrio por hilo, generadores aleatorios intactos e interruptor estricto |
| Mutaciones dirigidas (`nvtx_mutations.py`) | 8 de 8 detectadas (`evidence/nvtx-mutations.json`) |
| Coste por rango apagado (`nvtx_cost.py`) | 0,19 µs sobre una llamada vacía |
| Coste por rango encendido sin herramienta | 0,83 µs sobre una llamada vacía |

Un recorrido de mac_online abre 812 rangos y uno de M1 unos 2.480. Apagados cuestan unos 0,15 ms y 0,47 ms por recorrido, del orden de 10⁻⁵ del tiempo total. La dispersión de los recorridos completos (hasta un 48 % entre tandas) no permite verlo de extremo a extremo, así que la garantía de que no cambian nada es la paridad bit a bit y no el tiempo.

## Siguientes pasos que propone la medida

Ninguno se aplica en esta rama. Cada uno necesita su propia tarea, pruebas de paridad y medida antes y después, y su peso no equivale a la ganancia esperada.

1. Huella de las entradas en Titans-MAC (12 % a 15 %). La segunda huella de `verify` repite la primera sobre los mismos búferes de solo lectura dentro del mismo `_observe`. Habría que decidir qué garantía da esa repetición antes de quitarla o sustituirla.
2. Comprobaciones de finitud por tensor (12 % a 14 %). Agruparlas en menos lanzamientos sin retrasar el error más allá de la fase que lo produce.
3. JSON canónico en cada instante (4 % a 9 %). Guardar la forma canónica de los objetos inmutables que se vuelven a serializar.
4. Metadatos Parquet en la ordenación del índice (1,1 GB fuera de `memory_limit`). Revisar el tamaño de grupo de los archivos intermedios o el modo de lectura con el mismo orden y los mismos bits.
5. Códec episódico de M1 tras ce9542e8 (30 % de las muestras). Perfilar el `einsum` de `encode` cuando esa rama esté integrada.

## Reproducción

Las rutas locales se sustituyen por marcadores: `<vista>` es el manifiesto de la vista, `<trabajo>` el directorio de índices guardados, `<eventos>` los eventos de `benchmarks/chronological_events.py` y `<salida>` el destino.

```bash
# Tiempo de Python, sin pausa (repetir con --gil para la otra captura)
MEDOID_INDICES=<índices> MEDOID_WIDTHS=<anchuras> memslot gpu -- py-spy record --rate 100 \
  --threads --nonblocking --format raw --output <salida>-all.collapsed -- python \
  reports/engineering/profiling-20261010/replay_stored_events.py \
  benchmarks/chronological_replay.py <vista> <trabajo> \
  --family titans --arm titans_mac_online --option precision='"fp32_strict"' \
  --option block_rows=1024 --option accumulation_rows=1024 --predictor max_batch=1024 \
  --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8 --events <eventos>
python reports/engineering/profiling-20261010/pyspy_split.py --all <a1> <a2> --all-seconds <s1> <s2> \
  --gil <g1> <g2> --gil-seconds <t1> <t2> --output <reparto>.json

# Fases en la GPU
MARS_TITAN_NVTX=1 memslot gpu -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output <salida> python benchmarks/chronological_replay.py ...
nsys stats --report nvtx_pushpop_sum,nvtx_gpu_proj_sum,cuda_gpu_sum,cuda_api_sum --format csv \
  --output <salida> <salida>.nsys-rep

# Memoria: pico real y atribución
memslot suite --max 9G -- python reports/engineering/profiling-20261010/memory_profiles.py \
  index <vista> --index <índice> --output <resultados>.jsonl --label index-plain
ARROW_DEFAULT_MEMORY_POOL=system memslot suite --max 9G -- python -m memray run --native \
  --aggregate -o <captura>.bin reports/engineering/profiling-20261010/memory_profiles.py \
  index <vista> --index <índice> --output <resultados>.jsonl --label index-memray
python reports/engineering/profiling-20261010/memray_peak.py <captura>.bin <reparto>.json
python reports/engineering/profiling-20261010/duckdb_native_split.py <captura>.bin <duckdb>.json

# Rangos: paridad, coste y mutaciones
python benchmarks/chronological_replay.py ... --capture <dir>/branch-on.pt
python reports/engineering/profiling-20261010/compare_capture.py <dir> <paridad>.json develop:branch-off branch-off:branch-on
python reports/engineering/profiling-20261010/nvtx_cost.py <coste>.json
python reports/engineering/profiling-20261010/nvtx_mutations.py <mutaciones>.json
```

`memory_profiles.py read` y `validation` siguen el mismo patrón que `index`. La validación necesita `memslot gpu` porque copia la matriz a `cuda:0`.

```bash
# Matriz de ajuste de XGBoost sin rondas sobre una vista reducida (borrar la vista al terminar)
python reports/engineering/profiling-20261010/view_subset.py <vista> <subvista> --every 16
TMPDIR=<temporal> memslot gpu --max 8G -- python \
  reports/engineering/profiling-20261010/memory_profiles.py matrix <subvista>/manifest.json \
  --max-bin 64 --validation --output <resultados>.jsonl --label <etiqueta>
python reports/engineering/profiling-20261010/matrix_scaling.py <resultados>.jsonl <vistas>.json \
  reports/engineering/profiling-20261010/evidence/memory-runs.json \
  configs/baselines/historical-masked-campaign-execution.json <escala>.json
```

`<vistas>.json` reúne el `summary.json` de cada subvista bajo su nombre. Las páginas se escriben en `TMPDIR` y se borran al liberar la matriz.

## Archivos

| Archivo | Contenido |
|---|---|
| `pyspy_split.py` | Reparto del tiempo de Python por función a partir de dos capturas de py-spy |
| `memory_profiles.py` | Recorridos de índice, lectura, validación residente y matriz de ajuste con su pico de RSS y fase |
| `view_subset.py` | Vista con uno de cada N activos verificados, enlazados en duro |
| `matrix_scaling.py` | Ajuste del pico de la matriz frente a las filas y estimación para la ventana completa |
| `memray_peak.py` | Reparto del máximo de memray por función del proyecto, marco Python y asignador |
| `duckdb_native_split.py` | Reparto nativo de la memoria viva de la ordenación con DuckDB |
| `replay_stored_events.py` | Envoltorio que repite eventos guardados sin leer la vista |
| `compare_capture.py` | Comparación bit a bit de dos capturas del recorrido cronológico |
| `nvtx_cost.py` | Coste por llamada de los rangos apagados y encendidos |
| `nvtx_mutations.py` | Mutaciones dirigidas del módulo de rangos |
| `evidence/replay-runs.json` | Tiempos, caudal, memoria y muestras perdidas de cada recorrido |
| `evidence/pyspy-*.json` | Repartos de mac_online y M1 sin pausa y bloqueantes |
| `evidence/nsys-*.csv` | Resúmenes de Nsight Systems por rango, por núcleo y por llamada CUDA |
| `evidence/memory-runs.json` y `evidence/memray-*.json` | Picos medidos y repartos de memray |
| `evidence/memory-view.json` | Activos y filas de la vista de medida |
| `evidence/xgboost-matrix-runs.json`, `-views.json` y `-scaling.json` | Ejecuciones de la matriz de ajuste con sus trazas, vistas reducidas y estimación |
| `evidence/memray-matrix.json` | Reparto de memray de la matriz de 364.104 filas con 256 bins |
| `evidence/nvtx-parity.json`, `nvtx-cost.json` y `nvtx-mutations.json` | Paridad, coste y mutaciones de los rangos |

Las capturas en bruto (perfiles plegados, trazas de Nsight Systems y capturas de memray) no se versionan por su tamaño.

## Límites

- La máquina estaba compartida y la dispersión entre ejecuciones de develop llega al 22 % en mac_online y al 48 % en M1. Las sobrecargas se dan como intervalo de dos tandas y no como valor puntual.
- py-spy sin `--native` atribuye el tiempo nativo a la función de Python que lo llama. Separar dentro de libtorch lanzamientos, reserva de memoria y espera necesitaría `perf` o los contadores de Nsight Compute, que requieren permisos de administrador.
- Los rangos NVTX son marcas del anfitrión. La proyección en la GPU de Nsight Systems abarca desde el primer núcleo lanzado en el rango hasta el último e incluye los huecos.
- Las medidas de memoria cubren el 82,7 % de los activos de la ventana y deben repetirse sobre la v3.1 completa.
- La matriz de ajuste se midió hasta 739.214 filas y su pico para la ventana completa es una extrapolación lineal. La secuencia con la validación residente debe medirse sobre la v3.1 completa antes de entrenar, y las rondas de XGBoost no se han medido.
