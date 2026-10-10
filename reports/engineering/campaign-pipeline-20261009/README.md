# Tubería de la campaña: lectura, ranuras GPU y memoria

Fecha: 9 y 10 de octubre de 2026. Tarea [#363](https://github.com/GonxKZ/mars-titan/issues/363), rama `perf/campaign-pipeline`.

Este informe mide dónde espera la GPU durante la campaña con máscaras, qué ganan los cambios de lectura y de concurrencia de la rama y qué opciones de ejecución se recomiendan. No se ha ejecutado ningún entrenamiento, paso de optimizador ni evaluación científica. Las medidas en `cuda:0` recorren lotes reales con forward, pérdida y backward bajo el patrón sin pasos de `campaign_throughput` (`_forbid_steps`), y las de lectura solo leen. Todos los recibos están en [`measurements.json`](measurements.json).

## Condiciones

- **Equipo.** RTX 4070 Laptop (8 GiB), 16 hilos lógicos en 8 núcleos y un nodo NUMA, 32 GB de RAM y perfil `power-saver` sin cambios. PyTorch 2.14.0+cu130.
- **Aritmética.** FP32 estricto en todas las medidas GPU: TF32 desactivado en matmul y cuDNN, `float32_matmul_precision="highest"`, algoritmos deterministas y `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Los picos de VRAM se midieron con esa configuración.
- **Carga ajena.** Otras sesiones usaban CPU, disco y GPU a la vez (carga media de 13 a 22 con 16 hilos). Las cifras absolutas varían bastante entre repeticiones. Las comparaciones válidas son las de una misma serie, y las decisiones se apoyan en series repetidas y alternadas.
- **Datos.** Vista `US+CN/fold-012` de la edición con máscaras, la ventana más poblada. El lector cronológico se midió también en `CN/fold-012`. Casos `gru-00`, `transformer-00` y `dlinear-00` de la campaña A con su tamaño de lote declarado (256 filas).
- **Referencia.** Revisión `7a9e93e3` de `develop`, anterior a la rama.
- **Procedimiento.** Los subcomandos de [`benchmarks/campaign_pipeline.py`](../../../benchmarks/campaign_pipeline.py) reúnen los procedimientos de medida, y la revisión medida se fija con `PYTHONPATH`. Las cifras de este informe se tomaron con versiones previas de esos mismos procedimientos, fuera del repositorio. La versión versionada solo se ha comprobado con recorridos cortos.

## Resumen

| Medida | Antes (`7a9e93e3`) | Ahora |
| --- | --- | --- |
| Lector del corpus, un proceso, filas/s | 27.900 y 31.300 | 44.600 a 45.300 en serie, 92.600 a 98.900 con 2 a 8 hilos |
| Espera de la GPU al lector en el recorrido real | 60 % a 87 % del tiempo | 0,8 % a 14 % con 4 hilos y prefetch 4 |
| Tramo GPU del paso sobre el tiempo de pared | 14 % a 44 % | 86 % a 98 % |
| Apertura de `US+CN/fold-012` en un proceso nuevo | 1.415 s (SHA-256 de unos 54 GB) | 1,5 a 2,1 s con las huellas compartidas |
| Tres trabajos a la vez con MPS frente al mejor trabajo solo | (sin ranuras) | ×1,14 a ×1,88 en cinco series, mismos bits |
| Lector cronológico en los últimos instantes de `US+CN/fold-012`, filas/s | 36 con la LRU de 1 GiB, a mitad del ajuste (medida de `perf/campaign-kernels` en otros instantes) | 4.988 en serie, 6.281 a 9.711 con hilos, con 6,1 a 6,6 GiB de RSS |

El caudal de un trabajo de referencia pasa de 9.200 a 40.400 filas/s (GRU), de 17.900 a 33.300 (Transformer) y de 21.200 a 48.500 (DLinear) en la misma sesión. Tres trabajos con MPS suman entre 44.100 y 74.200 filas/s. Frente a las tres referencias de la revisión base ejecutadas una tras otra (14.200 filas/s de media armónica), el agregado es de ×3,1 a ×5,2, aunque esas cifras proceden de sesiones distintas y solo son orientativas.

La lectura ya no es el límite del recorrido de un trabajo. Con tres trabajos a la vez el límite pasa a ser la CPU del equipo, que comparten la lectura de cada trabajo y las cargas ajenas.

## Recorrido real: dónde espera la GPU

`campaign_pipeline.py steps` recorre 150 lotes tras 5 de calentamiento, con la apertura fuera de la medida. «Espera» es el tiempo del bucle bloqueado en `next()` sobre el lector. «Tramo GPU» suma el tiempo entre dos eventos CUDA registrados antes del forward y después del backward de cada lote. Incluye los huecos del stream dentro de ese tramo, así que es una cota superior de la ocupación real de la GPU, no una lectura de sus contadores.

| Caso | Configuración | Filas/s | Espera | Tramo GPU |
| --- | --- | --- | --- | --- |
| gru-00 | base | 9.205 | 86,6 % | 14,1 % |
| gru-00 | rama, 0 hilos | 7.758 | 89,0 % | 11,6 % |
| gru-00 | rama, 4 hilos y prefetch 4 | 40.447 | 14,1 % | 86,1 % |
| gru-00 | rama, 8 y 8 | 41.660 | 9,2 % | 90,4 % |
| transformer-00 | base | 17.893 | 60,0 % | 43,7 % |
| transformer-00 | rama, 0 hilos | 23.145 | 48,8 % | 55,5 % |
| transformer-00 | rama, 4 y 4 | 33.261 | 2,0 % | 98,2 % |
| transformer-00 | rama, 8 y 8 | 33.803 | 0,6 % | 99,6 % |
| dlinear-00 | base | 21.175 | 70,1 % | 28,7 % |
| dlinear-00 | rama, 0 hilos | 34.614 | 62,7 % | 36,6 % |
| dlinear-00 | rama, 4 y 4 | 48.498 | 0,8 % | 98,4 % |
| dlinear-00 | rama, 8 y 8 | 47.632 | 0,8 % | 98,4 % |

La primera configuración de la rama se midió justo después de la primera apertura, que había leído 54 GB y desplazado la caché de páginas, y eso explica que `gru-00` sin hilos quede por debajo de la base. Pasar de 4 a 8 hilos no mejora el caudal. El pico de RSS del proceso subió de 1,25 GB a 2,18 GB con la tubería.

Con la lectura resuelta, el Transformer y DLinear tienen la GPU ocupada casi todo el tiempo del paso. Lo que queda dentro de ese tramo (copias H2D, `.item()` por paso del entrenador y el propio cálculo) corresponde a los entrenadores y a sus núcleos, que trata la rama `perf/campaign-kernels`.

## Lector del corpus

`campaign_pipeline.py reader` abre la vista en un proceso nuevo y lee 400 lotes de entrenamiento (102.400 filas) sin modelo. Las dos revisiones tomaron las huellas SHA-256 del archivo compartido para no releer la vista, y la apertura queda fuera de la medida. Dos rondas alternadas:

| Configuración | Ronda 1 | Ronda 2 | CPU por fila |
| --- | --- | --- | --- |
| base | 27.887 | 31.337 | 32 a 36 µs |
| rama, en serie | 45.342 | 44.640 | 22 µs |
| rama, 2 hilos y prefetch 2 | 98.885 | 92.610 | 27 a 28 µs |
| rama, 4 y 4 | 94.772 | 96.986 | 29 µs |
| rama, 8 y 8 | 92.777 | 95.644 | 30 µs |

En serie, la lectura conjunta de los grupos de cada activo (`read_row_groups` con `pre_buffer`) baja el coste de 32 a 36 µs por fila a 22 µs. Los hilos reparten esa lectura y el montaje de los lotes, y más de dos no aportan en este equipo cargado. El pico de RSS del proceso fue de unos 0,8 GiB en la base y de 1,4 GiB en la rama, que recorre todas sus configuraciones.

La igualdad se comprueba en `tests/training/test_pipeline_batches.py` frente al lector de `14c6b1dd` cargado desde git: lotes, cursores, reanudación desde cada cursor, las tres ablaciones de modalidades y los fallos (vectores no finitos, disponibilidad futura, ventanas de precios inválidas y grupos vacíos) en la misma posición. Fuera de la edición con máscaras se conserva la lectura por grupo, para no retener activos completos.

## Huellas compartidas

Abrir una vista comprueba el SHA-256 de todos sus archivos. En la base, abrir `US+CN/fold-012` costó 1.415 s con el disco compartido, y cada proceso de una ranura habría repetido esa lectura. Con `MARS_TITAN_DIGEST_CACHE` las huellas se guardan en un archivo común con cerrojo y escritura atómica, y cada una solo vale para la ruta y la firma de stat exactas (dispositivo, inodo, tamaño, mtime y ctime). Un proceso nuevo abre la vista en 1,5 a 2,1 s. Un archivo reescrito cambia de ctime, se lee completo y se rechaza si su huella no coincide (prueba `test_shared_digests_spare_rehashing_and_still_detect_changes`).

## Lector cronológico

Las familias de memoria (Titans-MAC, MARS-TITAN y CM-v1) leen por instantes con `FinancialObservationSource.batched_events`. En `develop`, cada fila decodificaba otra vez su grupo Parquet salvo que siguiera en una caché LRU de 1 GiB. La rama `perf/campaign-kernels` midió ese lector en `US+CN/fold-012` a mitad del ajuste, con unos 3.240 flujos por instante: 36 filas/s, con 77.724 llamadas a `read_row_group` y 53.412 aperturas de archivo para 38.862 filas. Con 6 GiB de caché y bloques de 4.096 filas subió a unas 1.930 filas/s, tras 205 s para el primer instante. En sus recorridos reales sin pasos, el lector ocupaba entre el 83 % y el 97 % del tiempo de Titans y CM-v1.

En esta rama cada activo conserva su último grupo decodificado dentro de un presupuesto (4 GiB por defecto), los metadatos Parquet se leen una vez por activo y recorrido, los grupos de los cuatro instantes siguientes se decodifican antes en hilos y las ventanas de precios de un bloque se montan con una sola transformación vectorizada. Los grupos de muestras tienen 128 filas y ocupan unos 0,7 MB decodificados, así que un grupo por cada uno de los cerca de 5.000 activos de la ventana conjunta son unos 3,2 GiB y caben en el presupuesto.

Con un presupuesto menor que el conjunto activo, descartar el grupo usado hace más tiempo no reutiliza nada, porque cada instante recorre los activos en el mismo orden y el descartado es siempre el siguiente que se pide. La caché descarta ahora primero los grupos de activos ausentes durante 16 instantes y después los usados más recientemente. En una simulación del acceso cíclico con 1.400 grupos de sitio para 5.000 activos, la proporción de aciertos pasa de 0 a 0,27, y `test_a_cache_smaller_than_the_active_assets_still_reuses_groups` lo comprueba sobre el corpus de prueba con los mismos eventos.

Medidas de la rama (filas/s, bloques de 128 filas, sin GPU):

| Vista e instantes | En serie | 4 hilos y prefetch 4 | Otros |
| --- | --- | --- | --- |
| `CN/fold-012`, instantes 3.000 a 3.300 | 5.599 | 16.457 | 8 hilos: 13.752. 16 hilos: 14.220 |
| `US+CN/fold-012`, últimos 60 o 100 instantes | 4.988 (60) | 9.711 (100) | 2 hilos y prefetch 2: 6.281 (100) |

En `US+CN/fold-012` los últimos instantes tienen unos 2.400 flujos de entrada, con 4.946 activos y 7.259 grupos distintos en los últimos 110. Ningún grupo se decodificó dos veces. Las tres configuraciones se midieron seguidas con cargas de 19, 22 y 34, así que las diferencias entre ellas son orientativas. La carga mayor coincidió con 4 y 4, la configuración más rápida. Las vistas `CN` y `US` se retiraron antes de medir la base en las mismas condiciones, así que la comparación con `develop` se apoya en las cifras de `perf/campaign-kernels`.

**Memoria.** La primera medida en `US+CN/fold-012` superó los 9 GiB de su plaza y el sistema la terminó. El lector guardaba los metadatos Parquet de cada activo para no releer su pie, y en memoria ocupan unos 886 KiB por archivo (113 MiB serializados frente a 865 MiB retenidos en 1.000 archivos). Ahora solo guarda el número de grupos, y reabrir el pie cuesta unos 1,1 ms por grupo decodificado. Tras el cambio, el pico de RSS fue de 6,1 GiB en serie, 6,6 GiB con 2 y 2 y 6,2 GiB con 4 y 4, con 3,3 GiB de grupos en la caché, y la construcción del índice alcanzó 3,5 GiB antes de leer. En `CN/fold-012`, medido antes del cambio, el pico fue de 2,3 a 2,6 GB. `test_the_reader_keeps_only_the_group_count_of_each_file` comprueba que el lector no retiene ningún objeto de metadatos.

**Reanudación.** Empezar en un cursor avanzado recorre en Python todas las filas anteriores del índice. En el instante 9.286 de 9.356 eso tardó entre 6 y 18 minutos según la carga, antes de leer el primer instante.

## Ranuras GPU y MPS

`campaign_pipeline.py slots` lanza uno, dos o tres trabajos en procesos nuevos con el entorno de su ranura (`slot_environment`), 1.536 MiB de VRAM por proceso acotados con `bound_vram`, FP32 estricto y algoritmos deterministas. Cada trabajo recorre 600 lotes con forward, pérdida, backward y la comprobación `.item()` por paso del entrenador, y resume pérdidas y gradientes en una huella. Todos empiezan a la vez tras abrir su vista. El agregado son las filas de todos entre el primer inicio y el último final.

**En todas las series, cada trabajo acompañado dio la misma huella que solo.**

| Serie | Carga al empezar | Mejor solo | Uno tras otro | 2 a la vez | 3 a la vez | 3 frente al mejor solo |
| --- | --- | --- | --- | --- | --- | --- |
| Sin MPS, primera | 17,0 | 32.841 | 24.322 | 45.036 | 62.450 | ×1,90 |
| Con MPS, primera | 19,0 | 47.535 | 39.260 | 55.832 | 74.161 | ×1,56 |
| Con MPS (A1) | 14,3 | 40.330 | 31.188 | 47.819 | 68.250 | ×1,69 |
| Sin MPS (B) | 15,2 | 42.102 | 38.102 | 36.059 | 45.174 | ×1,07 |
| Con MPS (A2) | 17,0 | 38.714 | 32.976 | 39.388 | 44.141 | ×1,14 |

Filas por segundo. «Uno tras otro» es el caudal de los tres trabajos ejecutados en serie. Las series A1, B y A2 se ejecutaron seguidas para separar el efecto de MPS del orden y de la caché de páginas. En las series con MPS los trabajos solos también usan el demonio.

- Tres trabajos superan siempre al mejor trabajo solo, entre ×1,07 y ×1,90, y a los tres en serie, entre ×1,19 y ×2,57.
- Con MPS, dos trabajos nunca quedaron por debajo del mejor solo (×1,02 a ×1,19). Sin MPS, la serie B dio ×0,86 con dos trabajos.
- La diferencia entre series de un mismo modo (44.100 a 74.200 con MPS y tres trabajos) es mayor que la diferencia entre modos. Con estas cifras MPS no se puede declarar mejor en media, pero sí más estable con dos trabajos, y no empeoró ningún caso.

Tres ranuras con MPS y dos tuberías por trabajo, alternadas en una misma serie:

| Tubería por trabajo | Carga al empezar | Mejor solo | 2 a la vez | 3 a la vez |
| --- | --- | --- | --- | --- |
| 2 hilos y prefetch 2 | 14,3 | 43.607 | 56.640 | 70.250 |
| 4 hilos y prefetch 4 | 16,2 | 42.652 | 52.912 | 71.846 |
| 2 hilos y prefetch 2 | 17,8 | 45.137 | 55.672 | 79.720 |

Con tres trabajos, 2 y 2 rinde lo mismo que 4 y 4 dentro de la variación, con la mitad de hilos de lectura por trabajo. También aquí cada trabajo acompañado dio la misma huella que solo.

### CPU compartida

Con MPS y una carga sintética de procesos de Python puro ocupados al lado (`campaign_pipeline.py cpu-load`, sin memoria ni GPU), como la que tendría la etapa RL en CPU:

| Carga añadida | Solos (gru, transformer, dlinear) | 3 a la vez |
| --- | --- | --- |
| 8 procesos | 10.831, 9.468 y 12.189 | 22.858 |
| 4 procesos | 21.339, 23.096 y 28.268 | 48.783 |

Con ocho procesos ocupados, tres trabajos rinden entre la mitad y un tercio de lo medido sin esa carga. Con cuatro, el resultado queda dentro de la variación entre series. La lectura es el recurso compartido. Según la PR #432, la etapa RL rinde unas 9 veces más con 8 procesos CPU que en `cuda:0` de uno en uno, y entre 3,3 y 3,7 veces con 4 procesos bajo MPS. Lanzar la RL con 8 procesos CPU mientras la GPU entrena predictores costaría a estos más de la mitad de su caudal en este equipo. Con 4 procesos CPU, o con la RL bajo MPS en la GPU, el coste medido es menor. La decisión corresponde a la coordinación de la campaña.

## Memoria por familia

La VRAM declarada es la de todo el proceso, con unos 512 MiB de contexto de CUDA fuera del asignador. Las estimaciones proceden de medidas en la ventana más poblada. Si un trabajo terminado supera su estimación, su pico reservado más el contexto pasa a ser la estimación de su modelo, brazo y ámbito (`observed-resources.json`), y si agota su VRAM se repite con más reserva. Un modelo puede declarar también recursos por brazo (`arms`), porque los brazos de un mismo modelo pueden diferir mucho.

| Modelo | VRAM | RAM | Base de la estimación |
| --- | --- | --- | --- |
| Referencias neuronales | 1.024 MiB | 4.096 MiB | Pico reservado de 70 a 148 MiB en `US+CN/fold-012` (este informe) y RSS de 1,8 a 1,9 GiB por proceso con la tubería 4 y 4 |
| Ridge | 1.024 MiB | 3.072 MiB | Pico de 226 MiB y 1,5 GiB de RAM (informe `campaign-tabular-20261009` de la PR #433), más contexto y tubería |
| XGBoost | 7.000 MiB, en exclusiva | 8.192 MiB | 6.900 B por fila de validación, 17 B por fila de ajuste y 1,5 GiB (unos 5,6 GiB en `US+CN/fold-012`, PR #433). Sus rondas no se han medido |
| Titans-MAC, MARS-TITAN y CM-v1 | 1.792 MiB, CN 1.280 MiB | 9.216 MiB, CN 5.120 MiB | Grafo acumulado de `mac_online` con `accumulation_rows` 128: 1,22 GB por tramo con 5.023 flujos y 0,53 GB con 1.024 en `cuda:0` ([recibo](../cuda-checks-20261009/titans-chronological-memory-campaign-cuda.json)). RAM: pico de 6,6 GiB del lector cronológico en `US+CN/fold-012` (este informe), más contexto CUDA y entrenador |
| GRU candidata | 2.048 MiB, CN 1.024 MiB | 9.216 MiB, CN 5.120 MiB | Medidas de memoria del entrenador con acumulación ([recibo](../candidate-trainer-memory-20261009.json)). RAM: el mismo lector cronológico |

Sin acumulación, el grafo de un tramo de `mac_online` con todos los flujos ocuparía unos 14 GB y no cabe en la GPU. Con 128 filas por acumulación cabe y deja sitio para empaquetar. Un `accumulation_rows` mayor reduce pasadas pero aumenta la VRAM de cada trabajo, y con ella el número de trabajos que caben a la vez. Ese valor forma parte de la huella de cada receta y cambia el orden de las sumas, así que no se elige aquí por rendimiento. Lo fijan las ramas de cada familia antes de lanzar la campaña, con esta declaración actualizada a su pico medido. Con las recetas de `perf/campaign-kernels` (bloques de 1.024 filas), `perf/campaign-kernels` midió unos 5.900 MiB por proceso para `titans_mac_online` y los núcleos de CM-v1, y unos 2.400 MiB para `titans_mac_frozen`. Si se integran esas recetas, la declaración debe recoger esas cifras por brazo.

## Opciones recomendadas

`configs/baselines/historical-masked-campaign-execution.json`, que la campaña recibe con `run --execution`:

- **GPU.** Tres ranuras, 7.000 MiB de presupuesto y MPS en `/run/user/1000/nvidia-mps`. Cada trabajo acompañado da los mismos bits y el agregado supera al mejor trabajo solo en todas las series medidas. El presupuesto deja unos 1.100 MiB para el escritorio y el contexto del proceso de la campaña.
- **Tubería.** Dos hilos de decodificación y prefetch 2 por trabajo. Rinden lo mismo que 4 y 4 con tres ranuras y en el lector solo, y dejan más CPU para los demás trabajos.
- **CPU.** `threads_per_job` 2 y un trabajador CPU. Ningún trabajo largo de CPU con más de cuatro procesos en paralelo a las ranuras.
- **Modelos.** Las cifras de la tabla anterior. XGBoost y Ridge con `campaign_process`: se ejecutan de uno en uno en un hilo del proceso de la campaña, de modo que los ajustes de una ventana comparten la matriz cuantizada y las estadísticas de Ridge (PR #433) sin bloquear las ranuras. XGBoost declara todo el presupuesto, así que no comparte la GPU hasta que se midan sus rondas. Antes de cada lanzamiento, la campaña libera en su proceso la matriz y la Gram que el trabajo no usa. Las familias cronológicas declaran 9.216 MiB de RAM en `US+CN` por su lector.

## Pruebas y mutaciones

- Lectura: `test_pipeline_batches.py`, `test_pipeline_events.py`, `test_input_pipeline.py` y las pruebas del lector, la observación financiera y la ablación de modalidades (247 superadas y 1 omitida antes de integrar `develop`). Sobre el código final, `test_pipeline_batches.py`, `test_pipeline_events.py`, las pruebas de la fuente de observaciones financieras, `test_corpus_inputs.py` y `test_titans_walk_forward.py` dan 179 superadas y 1 omitida.
- Ranuras y campaña: `test_campaign_slots.py`, `test_masked_campaign.py` y `test_campaign_storage.py`. Concurrencia con los mismos recibos que en serie, pausa y reanudación de cada trabajo desde su intento, fallo de un proceso, trabajo bloqueado por otro proceso, VRAM no ampliable, reintento tras agotar la VRAM, estimaciones observadas, orden de lanzamiento con reserva del primer trabajo, ejecución en el proceso de la campaña y lectura del informe de procesos de la GPU. Con `test_gpu_supervisor.py`, `test_input_pipeline.py` y `test_tabular_shared.py`, 175 superadas sobre el código final tras integrar `develop`.
- Una ejecución de `test_campaign_process_jobs_run_there` falló porque un proceso de ranura envió su resultado y terminó entre las dos comprobaciones de la campaña, que miraba la tubería antes que el estado del proceso y lo dio por perdido. Ahora lee primero el estado y recoge el resultado en la vuelta siguiente, y `test_a_result_sent_just_before_the_process_exits_is_not_lost` reproduce ese orden.
- Una ejecución de la prueba de grupos corruptos se bloqueó. El volcado de los hilos mostró que el recolector de basura había finalizado un recorrido abandonado (el de la pasada anterior, que terminó con el error) dentro de una de sus propias tareas de decodificación, y que el cierre esperaba a esa misma tarea. Ahora `drain` solo cancela cuando se ejecuta dentro de una tarea, `_blocks` cierra su recorrido en su propio hilo tras un error y `test_a_map_closed_inside_its_own_task_does_not_wait_for_itself` reproduce el caso en un proceso aparte (sin la corrección agota su tiempo).
- 56 mutaciones dirigidas distintas, aplicadas de una en una, en `measurements.json`, entre ellas las de los recursos por brazo (B1 a B5), el orden de comprobación de las ranuras (P1), la retención de metadatos Parquet (R1) y la integración con la PR #433 (T1 a T3). Todas hacen fallar alguna prueba salvo tres equivalentes: desplazar el cursor consumido en el plan (M5), que el plan recalcula, repartir en hilos huellas que ya están en memoria (N14), que `_digest` no vuelve a calcular, y leer los grupos de un activo en orden inverso (A5), porque los tramos se calculan con el mismo orden. Las que sobrevivieron en una primera ronda (M4, M10, M11, N11, A6 y la política de la caché cronológica) dieron lugar a pruebas nuevas.

## Pendiente

- **Control por caudal medido.** El número de ranuras es una decisión declarada con estas medidas. La campaña no mide todavía el caudal agregado para añadir o quitar ranuras, porque los entrenadores no publican su progreso por filas.
- **Rondas de XGBoost y familias cronológicas en `cuda:0`.** Sus picos reales se incorporarán a `observed-resources.json` en la primera ejecución y a la declaración cuando se midan sin el bloqueo.
- **Memoria fijada, stream de copia y sincronizaciones por paso.** Corresponden a los entrenadores (`perf/campaign-kernels`).
- **Edición v3.1 (PR #445).** Sus ventanas de precios con canal de presencia sustituyen a `_price_contexts` dentro de `corpus_inputs.py`. Al integrar las dos ramas, la ruta por activo y las ventanas vectorizadas deben llamar a `price_windows`, y las pruebas de igualdad se repetirán con esa edición.
- **Reanudación cronológica.** Saltar hasta un cursor avanzado recorre en Python todo el índice anterior (6 a 18 minutos al final de `US+CN/fold-012`). Una conciliación vectorizada por grupos Parquet del índice lo evitaría sin perder la comprobación del resumen.
- **Escalas de M3.** `fit_write_scalers` recorre todo el tramo de ajuste dentro del trabajo GPU de M3 antes de su primera actualización, y la ranura queda reservada mientras solo lee. Separarlo en un trabajo CPU previo es un cambio del plan.
- **Repetición sin carga ajena.** Las series se midieron con otras sesiones activas. Antes de la campaña conviene repetir `campaign_pipeline.py slots` con el equipo dedicado.
