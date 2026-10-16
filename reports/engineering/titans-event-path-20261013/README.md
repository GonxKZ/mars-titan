# Trabajo de anfitrión por evento de Titans-MAC

Recibos del 13 al 16 de octubre de 2026 para [#549](https://github.com/GonxKZ/mars-titan/issues/549). La descripción de la vista CPU sellada y de las comprobaciones diferidas está en el [adaptador financiero](../../../docs/engineering/titans-financial-adapter.md) y en el [entrenador cronológico](../../../docs/engineering/titans-chronological-trainer.md).

| Archivo | Contenido |
| --- | --- |
| [`parity.json`](parity.json) | Igualdad de bits entre `develop` y la rama en CPU y `cuda:0`, con FP32 estricto y FP64 |
| [`work-counts.json`](work-counts.json) | Trabajo de anfitrión contado en un recorrido de `mac_online` y en uno del lector M1, antes y después |
| [`mutations-cpu.json`](mutations-cpu.json), [`mutations-cuda.json`](mutations-cuda.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`parity_capture.py`](parity_capture.py), [`parity_compare.py`](parity_compare.py) | Captura y comparación bit a bit |
| [`work_counts.py`](work_counts.py) | Cuentas de SHA-256, JSON, sincronizaciones y lanzamientos |
| [`mutations.py`](mutations.py) | Mutaciones, cada una con sus pruebas |
| [`abba-20261014.json`](abba-20261014.json) | Rondas A B B A del recorrido con eventos reales |
| [`profiles-20261016.json`](profiles-20261016.json), [`profiles.py`](profiles.py) | Perfiles de py-spy y Nsight Systems del recorrido y benchmark del CUDA Graph de la emisión |

Ningún recibo ejecuta pasos de optimizador. Los recorridos usan `RecordingOptimizer` de las pruebas, que no hereda de `torch.optim.Optimizer` y solo copia los gradientes de cada llamada, con el gancho de `campaign_throughput` que prohíbe cualquier paso de `torch.optim`. Al final se comprueba que ningún parámetro ni búfer ha cambiado. La protección local de aprendizaje sigue activa.

## Qué cambia

1. **Vista CPU sellada.** `validated_cpu_batch` copia cada arreglo una sola vez a un objeto `bytes` y conserva su vista de solo lectura (antes había una copia y después otra con `tobytes`). La huella SHA-256 se calcula al crear la vista, directamente sobre esos objetos. `CPUDecisionBatch.verify` ya no la recalcula. Exige los mismos objetos que se sellaron, con la misma forma, tipo y pasos, sin permiso de escritura y respaldados por `bytes`. Como un objeto `bytes` no se puede modificar y NumPy no deja volver a hacer escribible una vista suya, eso basta para saber que los datos no han cambiado.
2. **Comprobaciones de finitud diferidas por evento.** `_observe` abre un tramo de `deferred_checks` por evento y `_replay` uno por grupo de flujos, en Titans-MAC y en el lector. En CUDA todas las comprobaciones de los bloques del tramo se resuelven con una sola espera al final, con el mismo `ValueError` y el mismo mensaje que la comprobación inmediata. El indicador de un tensor real es su norma infinito, una sola reducción sin temporales, que es finita si y solo si lo son todos sus valores. Las dos comprobaciones del codificador de precios, que antes esperaban a la GPU dentro de `prepare`, entran en el mismo tramo. Los tramos anidados olvidan al cerrarse los tensores que recordaban para no repetir comprobaciones, así que un evento largo no retiene los temporales de cada bloque.
3. **Huellas calculadas una vez.** `MemoryConfig`, `MACConfig` y `FinancialInputSpec` guardan su huella al pedirla por primera vez, sin serializarla. `FinancialPredictor._config_id` la guarda por modo train/eval y la olvida al volver a sellar los parámetros (carga, cambio de tipo o de dispositivo).
4. **Identidad de decisión contra un único corte.** El bloque tiene un solo corte, así que cada fila se compara con ese instante y su sufijo, con la expresión del identificador de flujo compilada una vez.
5. **Lectura de las comprobaciones de un CUDA Graph.** `_PendingChecks.first_failure` lee los indicadores sin descartarlos y `resolve` la usa y después los descarta. `benchmarks/titans_emission_graph.py` (de [#565](https://github.com/GonxKZ/mars-titan/pull/565)) leía los indicadores como booleanos, lo que con las normas habría dado por buena una norma infinita y por fallida una norma cero. Ahora usa `first_failure` y las mismas comprobaciones del codificador que `prepare`.

## Garantías

- Una entrada sustituida por otra con los mismos valores, escrita, con otra forma, otro tipo o una copia escribible se rechaza antes de usarla (`test_cpu_view_with_a_replaced_array_is_rejected_even_with_the_same_values`, `test_cpu_view_rejects_a_writable_copy_and_a_changed_digest` y las pruebas de contrato de `tests/models/titans/test_financial_adapter.py`).
- La huella distingue una ausencia de un cero observado (`test_cpu_observation_digest_distinguishes_missing_from_observed_zero`).
- Un valor no finito detiene el cálculo con el mensaje de la primera comprobación fallida, en CPU en su línea y en CUDA al final del tramo, antes de devolver nada (`tests/models/titans/test_deferred_checks.py`, `test_price_encoder_checks_keep_their_message_inside_prepare` y las pruebas de finitud del lector).
- El indicador coincide con `isfinite(...).all()` para cada tipo real, con NaN, infinitos, ceros con signo, subnormales, máximos finitos, tensores vacíos y vistas con pasos, y no crea grafo de autograd (`test_finite_indicator_agrees_with_isfinite`, `test_finite_indicator_adds_no_autograd_graph`).
- Cada evento comparte un único tramo para todos sus bloques y la repetición uno por grupo de flujos (`test_each_event_resolves_the_checks_of_all_its_blocks_together`, `test_replay_resolves_the_checks_of_each_flow_group_together`, `test_each_event_resolves_core_and_reader_checks_together`).

## Igualdad de bits

`parity_capture.py` recorre con el código de cada árbol el corpus técnico de las pruebas (`tests/training/chronological_fixture.py`, `hidden_size` 32 y bloques de 2 filas), con FP32 estricto (sin TF32) y sin fastpath de atención:

- `ChronologicalTrainer.run()` de las cuatro variantes de Titans-MAC (`transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`), con una época de ajuste y sus validaciones,
- el tramo de ajuste del lector M0, M1 y M3 sobre el padre `mac_online` con cabeza de cuantiles.

Guarda la auditoría completa (cada predicción, etiqueta y pérdida en orden), los gradientes de cada llamada al optimizador sustituto, los estados de todos los flujos exportados a CPU antes de cada cierre de recorrido, la identidad de la ejecución y las métricas. `parity_compare.py` exige igualdad exacta de cada flotante y `torch.equal` con el mismo tipo y forma en cada tensor. La identidad se compara sin el SHA-256 de los módulos, que cambia con el código medido.

| Caso | Configuraciones | Predicciones por configuración | Llamadas al optimizador sustituto | Resultado |
| --- | --- | --- | --- | --- |
| CPU, FP64 | 4 variantes y 3 lectores | 302 (Titans-MAC) y 88 (lectores) | 11 | todo igual |
| CPU, FP32 estricto | 4 variantes y 3 lectores | 302 y 88 | 11 | todo igual |
| `cuda:0`, FP32 estricto | 4 variantes y 3 lectores | 302 y 88 | 11 | todo igual |
| `cuda:0`, FP64 | 4 variantes y 3 lectores | 302 y 88 | 11 | todo igual |

En los 28 pares coinciden la auditoría (605 entradas en Titans-MAC, 183 en M0 y 214 en M1 y M3), los gradientes de las 11 llamadas, los estados exportados en cada cierre (tres recorridos en Titans-MAC y uno en el lector), las métricas y la identidad. Los módulos cuyo SHA-256 cambia son los siete de la rama: `config`, `episodic_readout`, `financial`, `financial_inputs` y `state` de `models.titans`, y `financial_run` y `mars_titan_run` de `training`. Como control de que la captura distingue dispositivos, la captura de CPU y la de `cuda:0` de un mismo árbol en FP32 no coinciden (en M1, el gradiente difiere hasta 3,5e-10).

Los commits comparados son `develop` en 7a82b055 y la rama en 6a013588. Antes de integrar #565 la misma comparación, entre b9b1f36d y la rama sobre él, dio el mismo resultado. Esta igualdad está comprobada en el corpus técnico. La comparación con los eventos reales de la campaña está en [Medidas de tiempo](#medidas-de-tiempo).

## Mutaciones

`mutations.py` cambia un único fragmento del código de la rama, ejecuta las pruebas de esa garantía con `-x` y sin orden aleatorio y restaura el archivo. Con `cpu` se ocultan las GPU. Las seis mutaciones de CUDA necesitan el dispositivo porque solo allí se difieren las comprobaciones. Las 24 hacen fallar la prueba esperada sobre el commit 6a013588, y esas mismas pruebas pasan sin mutar.

| Mutación | Dispositivo | Qué rompe | Prueba que falla |
| --- | --- | --- | --- |
| `verify_without_identity` | CPU | `verify` deja de exigir los mismos arreglos sellados | `test_financial_adapter.py::test_cpu_view_with_a_replaced_array_is_rejected_even_with_the_same_values[prices]` |
| `verify_without_digest` | CPU | `verify` deja de comparar la huella guardada | `test_financial_adapter.py::test_cpu_view_rejects_a_writable_copy_and_a_changed_digest` |
| `digest_without_presence` | CPU | la huella omite la presencia | `test_financial_adapter.py::test_cpu_observation_digest_distinguishes_missing_from_observed_zero` |
| `inputs_not_frozen` | CPU | las entradas se conservan en copias escribibles | `test_financial_adapter.py::test_each_control_advances_one_decision_without_mutating_received_state[transformer_direct]` |
| `identity_without_suffix` | CPU | el identificador de muestra no necesita el sufijo del corte | `test_financial_adapter.py::test_cpu_boundary_names_each_identity_fault[other_cut]` |
| `identity_availability_lower` | CPU | se admite una disponibilidad negativa | `test_financial_adapter.py::test_cpu_boundary_names_each_identity_fault[negative_availability]` |
| `identity_cut_range` | CPU | se admite un corte fuera del intervalo histórico | `test_financial_adapter.py::test_historical_batch_rejects_incompatible_inputs_before_tensor_conversion[reserved]` |
| `indicator_l2_norm` | CPU | norma 2 en lugar de norma infinito | `test_deferred_checks.py::test_finite_indicator_agrees_with_isfinite[dtype0]` |
| `indicator_complex_norm` | CPU | los complejos pasan por la norma | `test_deferred_checks.py::test_finite_indicator_keeps_the_general_check_for_other_types[cpu]` |
| `indicator_keeps_graph` | CPU | la norma conserva el grafo de autograd | `test_deferred_checks.py::test_finite_indicator_adds_no_autograd_graph` |
| `observe_without_event_scope` | CPU | `_observe` de Titans-MAC sin tramo por evento | `test_financial_run.py::test_each_event_resolves_the_checks_of_all_its_blocks_together` |
| `replay_without_group_scope` | CPU | `_replay` sin tramo por grupo de flujos | `test_financial_run.py::test_replay_resolves_the_checks_of_each_flow_group_together` |
| `reader_without_event_scope` | CPU | `_observe` del lector sin tramo por evento | `test_mars_titan_run.py::test_each_event_resolves_core_and_reader_checks_together` |
| `zero_fill_on_present_rows` | CPU | el relleno cero se exige en las filas presentes | `test_episodic_readout.py::test_read_matches_a_two_neighbor_softmax_and_residual_equation` |
| `encoder_message_swapped` | CPU | los mensajes del codificador en orden inverso | `test_financial_adapter.py::test_price_encoder_checks_keep_their_message_inside_prepare[prices]` |
| `config_id_ignores_mode` | CPU | la huella del predictor ignora el modo train/eval | `test_financial_adapter.py::test_predictor_identity_follows_mode_dtype_and_loads` |
| `seal_keeps_config_ids` | CPU | volver a sellar no olvida las huellas guardadas | `test_financial_adapter.py::test_predictor_identity_follows_mode_dtype_and_loads` |
| `fingerprint_serialized` | CPU | la huella guardada se serializa con la configuración | `test_financial_adapter.py::test_configuration_fingerprints_are_computed_once_and_not_serialized` |
| `resolve_keeps_flags` | CUDA | `resolve` no descarta los indicadores leídos | `test_deferred_checks.py::test_first_failure_reads_without_discarding_and_resolve_discards` |
| `first_failure_discards` | CUDA | `first_failure` descarta los indicadores | `test_deferred_checks.py::test_first_failure_reads_without_discarding_and_resolve_discards` |
| `resolve_ignores_bounds` | CUDA | las normas se leen como ciertas sin comprobar su finitud | `test_deferred_checks.py::test_cuda_checks_resolve_at_the_end_with_the_first_message` |
| `nested_scope_keeps_seen` | CUDA | un tramo anidado conserva los tensores recordados | `test_deferred_checks.py::test_nested_block_forgets_its_tensors_but_keeps_their_checks` |
| `encoder_checks_immediate` | CUDA | el codificador vuelve a esperar a la GPU en su línea | `test_financial_adapter.py::test_cuda_prepare_waits_for_the_device_once[False]` |
| `readout_without_scope` | CUDA | la lectura episódica sin tramo propio | `test_episodic_readout.py::test_cuda_readout_waits_for_the_device_once[1]` |

## Trabajo contado

`work_counts.py` recorre en `cuda:0` y FP32 estricto `ChronologicalTrainer.run()` de `mac_online` (302 predicciones) y el tramo de ajuste del lector M1 (88 predicciones) sobre el corpus técnico, y cuenta el trabajo de anfitrión del recorrido sin la preparación de la fixture. No cronometra nada. Son cuentas de un recorrido determinista, así que la carga ajena de la GPU y de la CPU no las cambia. Las sincronizaciones salen de `torch.cuda.set_sync_debug_mode("warn")`, que PyTorch declara en prototipo, y coinciden con las llamadas a `cudaStreamSynchronize` del perfilador de PyTorch.

| Recuento | `mac_online`, develop | `mac_online`, rama | Lector M1, develop | Lector M1, rama |
| --- | --- | --- | --- | --- |
| Llamadas a SHA-256 | 5.153 | 1.518 | 1.517 | 560 |
| Bytes resumidos con SHA-256 | 518.646.058 | 514.298.622 | 21.797.033 | 20.512.883 |
| Llamadas a `json.dumps` | 4.820 | 1.185 | 1.834 | 877 |
| Caracteres de `json.dumps` | 6.856.302 | 3.069.694 | 1.969.718 | 952.670 |
| Sincronizaciones con el anfitrión | 1.676 | 1.250 | 2.176 | 345 |
| `cudaLaunchKernel` | 92.718 | 55.388 | 40.836 | 24.888 |
| `cudaMemcpyAsync` | 7.775 | 6.716 | 4.117 | 1.976 |
| `cudaStreamSynchronize` | 1.675 | 1.249 | 2.176 | 345 |

En `mac_online` los lanzamientos de núcleos bajan de 307 a 183 por predicción y las sincronizaciones de 5,5 a 4,1. En el lector M1 las sincronizaciones bajan de 24,7 a 3,9 por predicción y los lanzamientos de 464 a 283. Las llamadas a SHA-256 caen un 71 % y un 63 %, y las serializaciones JSON un 75 % y un 52 %.

Los bytes resumidos apenas cambian porque casi todos vienen de otro sitio. Un recuento por función llamante en la rama atribuye 506.207.683 bytes en 29 llamadas a `triton_key` de la caché de Triton, igual en `develop`. La huella de las vistas CPU, que ahora se calcula una vez por bloque en lugar de dos, suma 0,67 MB en este recorrido. El mismo recuento muestra 738 llamadas a `_tensor_digest` desde `_parameter_digest`, cada una con su copia al anfitrión. Es la frontera fuerte de identidad de parámetros y queda fuera de este cambio.

La fixture usa bloques de 2 filas y `hidden_size` 32, mientras la campaña usa bloques de cientos de filas y `hidden_size` 64. Estas proporciones muestran qué trabajo se ha retirado, pero no predicen la reducción de tiempo en la campaña, que solo puede salir de la medida del recorrido real.

## Medidas de tiempo

Las rondas se tomaron el 14 de octubre de 2026, con la GPU sin otras cargas de cómputo. La única carga ajena era un renderizador de Chrome, en torno al 48 % de un núcleo, y la carga media de cada ronda queda en el recibo [`abba-20261014.json`](abba-20261014.json).

- **Datos.** Vista `US+CN/fold-012` de las vistas v3.1 de A v2. Los eventos se guardaron una sola vez desde `develop` en 06e69db6, con `benchmarks/chronological_events.py VISTA TRABAJO eventos.pkl 100 6`, y empiezan en el evento 3.224 de 6.449. Los dos árboles recorren el mismo tramo de ajuste: 88.028 filas, 87.202 etiquetas y 11 llamadas a `_NoStepOptimizer`, que nunca aplica un paso.
- **Árboles.** A = 7012da85 y B = 37f4b7a7, el commit de #589 en `develop`. Entre los dos entran también [#594](https://github.com/GonxKZ/mars-titan/pull/594), que cambia cómo se trocean las tablas acotadas de `data/batches.py`, y dos cambios nativos de la evaluación de políticas. Ninguno toca el recorrido medido: los eventos vienen guardados, la lectura de datos ocupa entre el 0,03 % y el 0,11 % del tiempo en los dos árboles y el enlace episódico es el mismo binario, compilado desde `develop` en df7bb523.
- **Orden.** A B B A por modelo. Cada ronda va en su propio proceso y trabajo `memslot gpu`, con `--strict --warmup 2 --segments 8`, FP32 estricto (sin TF32 en matmul ni en cuDNN), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dos hilos y el perfil de energía de ahorro.
- **Opciones.** Titans-MAC usa la receta de la campaña. Los lectores usan además `precision="fp32_strict"`, `block_rows=512`, `max_batch=1024` y `max_state_bytes=134217728`, y las escalas de M3 salen de los mismos eventos en los dos árboles.

| Modelo | Filas/s antes | Filas/s después | Cociente de medias | Peor par | Pico asignado antes y después |
| --- | --- | --- | --- | --- | --- |
| `titans_mac_online` | 13.583 y 14.379 | 19.534 y 19.319 | 1,39 | 1,34 | 2.113 MiB en las dos |
| `mars_titan_m1` | 6.112 y 6.203 | 7.201 y 7.078 | 1,16 | 1,14 | 397 y 402 MiB |
| `mars_titan_m3` | 5.027 y 5.463 | 6.112 y 6.230 | 1,18 | 1,12 | 397 y 402 MiB |

El peor par divide la ronda más lenta de después entre la más rápida de antes. En los lectores el pico asignado sube 5 MiB (1,3 %) y el reservado de 480 a 492 o 494 MiB. En Titans-MAC no cambia.

Las capturas de las cuatro rondas de cada modelo coinciden bit a bit, entre árboles (rondas 1 y 2, 4 y 3) y dentro de cada árbol (1 y 4, 2 y 3). Coinciden los nombres de los parámetros, los gradientes de las 11 llamadas (442 tensores en Titans-MAC y 77 en cada lector) y las 3.347 predicciones pendientes al terminar, con `torch.equal` y el mismo tipo. Así queda comprobada con eventos reales la igualdad que [Igualdad de bits](#igualdad-de-bits) comprobó en el corpus técnico.

Como los eventos vienen guardados, los cocientes describen el recorrido por evento. El tiempo de una ventana completa incluye también la lectura de la vista, la preparación y las validaciones, y no se extrapola de estas rondas.

## Perfiles

Los perfiles se tomaron el 16 de octubre de 2026 entre las 04:03 y las 04:12, con el árbol de `main` en af476f38 (el mismo contenido que `develop` en e01a17b8, que ya integra #589), los mismos eventos y opciones de [Medidas de tiempo](#medidas-de-tiempo) y un trabajo `memslot gpu` por ejecución. El llama-server de Kaizten no usó la GPU durante los dos perfiles. La carga media estuvo entre 3,8 y 4,9. El recibo es [`profiles-20261016.json`](profiles-20261016.json), generado con [`profiles.py`](profiles.py).

Los dos perfiladores alargan el lado del anfitrión. El tramo medido tardó 7,79 s con py-spy y 9,26 s con Nsight Systems, frente a unos 4,5 s en las rondas limpias del 14 de octubre. Por eso los perfiles sirven para repartir el tiempo, pero no como medida de caudal.

| Parte del recorrido (py-spy, muestras inclusivas de `_train_pass`) | Fracción |
| --- | --- |
| `_observe`, la emisión de cada evento | 52,9 % |
| `_update`, con `_replay` (repetición del tramo con autograd) en 41,0 % | 43,8 % |
| `prepare` del predictor, en la emisión y en la repetición | 48,0 % |
| `forward` de `TitansMAC` | 20,5 % |
| `validated_cpu_batch`, la vista CPU sellada | 17,1 % |
| SHA-256 de la vista CPU (`_digest`) | 9,7 % |
| Actualización de la memoria neuronal (`NeuralMemory.update`) | 8,6 % |
| `encode_last` del codificador de precios | 7,6 % |
| `json.dumps` de `canonical`, sobre todo para contar los bytes de metadatos del estado | 4,4 % |
| `validate_historical_vectors` | 4,0 % |
| Motor de autograd en el backward | 3,4 % |
| `_emit`, el registro de las predicciones emitidas | 1,7 % |

En Nsight Systems, la ventana del recorrido, del primer evento a la última actualización, dura 14,58 s y contiene 207.328 núcleos con 2,84 s de cálculo en la GPU. La GPU solo está ocupada el 19,5 % de la ventana. Hay 197.371 llamadas a `cudaLaunchKernel` y 9.760 copias. La traza alarga el anfitrión, pero no los núcleos, así que sin ella la fracción ocupada sería mayor. Escalando la ventana por el tramo medido (9,26 s con la traza frente a 4,5 s sin ella) quedaría en torno al 40 %, una estimación y no una medida. El recorrido sigue limitado por el anfitrión, que lanza núcleos pequeños de unos 14 µs de media. La atención de PyTorch (`fmha_cutlass`) se lleva el 28 % del tiempo de los núcleos.

### CUDA Graph de la emisión

`benchmarks/titans_emission_graph.py` repite 20 veces, tras 3 de calentamiento, la emisión de un bloque de 1.024 flujos con `differentiable=True` por tres rutas: `prepare` con sus validaciones, el mismo cálculo sin ellas y ese cálculo capturado en un CUDA Graph. Las tres rutas dieron los mismos bits en las tres ejecuciones.

| Ejecución | `prepare` | Sin validaciones | CUDA Graph | Grafo frente al mismo cálculo sin grafo |
| --- | --- | --- | --- | --- |
| 1 | 8,74 ms | 6,44 ms | 6,38 ms | 1,01 |
| 2 | 14,88 ms | 12,46 ms | 12,95 ms | 0,96 |
| 3 | 9,61 ms | 7,46 ms | 6,26 ms | 1,19 |

Son medianas por ejecución. En la segunda el llama-server de Kaizten usó la SM en 4 de sus 11 muestras.

## Decisión

- **CUDA Graphs por forma exacta: se descartan.** El grafo no gana de forma reproducible al mismo cálculo sin él (1,01 de mediana, entre 0,96 y 1,19). La emisión dentro de `_observe` (`titans.prepare`, 182 bloques) ocupa 3,16 s de la ventana de 14,58 s, un 21,6 %, así que incluso con la mejor ejecución el ahorro no pasaría del 3,5 % del recorrido. Además, cada grafo solo sirve para una forma de bloque y los bloques cambian de tamaño entre eventos. La diferencia entre `prepare` y el cálculo sin validaciones (de 1,19 a 1,36) no está disponible, porque esas validaciones son las garantías de [Garantías](#garantías).
- **Núcleo CUDA propio para la actualización de memoria: se descarta mientras se exija igualdad de bits.** La actualización de la memoria neuronal ocupa el 8,6 % de las muestras del anfitrión, lo que acota el ahorro posible. Fusionarla cambiaría el orden de las operaciones respecto de la referencia de PyTorch y necesitaría su propio backward para la repetición del tramo.
- **Lo que queda medido.** Los costes mayores del anfitrión que no son cálculo del modelo son el SHA-256 de la vista CPU (9,7 %, ya con SHA-NI a 1,76 GB/s en OpenSSL 3.5.8), la serialización JSON que solo cuenta los bytes de metadatos del estado en `_usage` y `_state_size` (3,5 %) y `validate_historical_vectors` (4,0 %). Son garantías que deben conservarse, así que solo cabe calcularlas con menos trabajo o en paralelo, con los mismos resultados.

## Reproducción

- Pruebas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/models/titans tests/training/test_financial_run.py tests/training/test_mars_titan_run.py tests/test_learning_hold_lowering.py tests/training/test_learning_hold_inventory.py`. Con `cuda:0` visible pasaron 771 pruebas en 8 min 20 s, con 2,1 GiB de memoria residente máxima. Antes de integrar #565 pasaron también los archivos de campaña y validación por ventanas de Titans-MAC y MARS-TITAN (157 pruebas y una omitida). `tests/training/test_titans_campaign.py` necesita por sí solo 5,4 GiB de memoria residente en `develop`, así que conviene ejecutarlo en un proceso aparte si el límite es de 6 GiB.
- Igualdad de bits: `python parity_capture.py ÁRBOL salida.pt DISPOSITIVO DTYPE TRABAJO` en cada árbol, con el mismo `TRABAJO` en los dos porque las huellas del corpus incluyen su ruta, y después `python parity_compare.py develop.pt rama.pt`.
- Trabajo contado: `python work_counts.py ÁRBOL salida.json TRABAJO`.
- Mutaciones: `python mutations.py ÁRBOL salida.json cpu` y `... cuda`.
- Eventos, una vez y sin GPU: `python benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`.
- Rondas alternas, cada una en su propio trabajo `memslot gpu`: `python benchmarks/chronological_replay.py VISTA/manifest.json TRABAJO --family titans --arm titans_mac_online --strict --warmup 2 --segments 8 --events eventos.pkl --capture CAPTURA.pt --output resultados.jsonl --label ETIQUETA`. Para los lectores, `--family readout --arm mars_titan_m1` y `--arm mars_titan_m3` con `--option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728`, y `BENCH_M3_EVENTS=eventos.pkl` para que M3 estime sus escalas con los mismos eventos. Bits de las rondas: `python reports/engineering/profiling-20261010/compare_capture.py DIRECTORIO salida.json antes:despues`.
- Perfiles, sobre la orden de la ronda de `titans_mac_online` sin `--capture`: `py-spy record --rate 100 --format raw -o DIRECTORIO/titans.pyspy -- python ...` y `MARS_TITAN_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o DIRECTORIO/titans python ...`, después `nsys export --type sqlite -o DIRECTORIO/titans.sqlite DIRECTORIO/titans.nsys-rep`. La emisión con CUDA Graph: `python benchmarks/titans_emission_graph.py eventos.pkl VISTA/manifest.json DIRECTORIO/graph-N.json 1024`. Cada ejecución guarda `nvidia-smi pmon -s u -d 1` en `DIRECTORIO/ETIQUETA.pmon`. El recibo sale de `python profiles.py DIRECTORIO salida.json` desde la raíz del repositorio.

Todas las órdenes usan `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dos hilos y `PYTHONPATH=ÁRBOL/src`, y las del lector el enlace episódico de `MARS_TITAN_EPISODIC_NATIVE`. Las pruebas, capturas y recuentos se ejecutaron en una RTX 4070 Laptop con torch 2.14.0 y CUDA 13.0.
