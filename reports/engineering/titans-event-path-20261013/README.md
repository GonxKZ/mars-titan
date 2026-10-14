# Trabajo de anfitrión por evento de Titans-MAC

Recibos del 13 de octubre de 2026 para [#549](https://github.com/GonxKZ/mars-titan/issues/549). La descripción de la vista CPU sellada y de las comprobaciones diferidas está en el [adaptador financiero](../../../docs/engineering/titans-financial-adapter.md) y en el [entrenador cronológico](../../../docs/engineering/titans-chronological-trainer.md).

| Archivo | Contenido |
| --- | --- |
| [`parity.json`](parity.json) | Igualdad de bits entre `develop` y la rama en CPU y `cuda:0`, con FP32 estricto y FP64 |
| [`work-counts.json`](work-counts.json) | Trabajo de anfitrión contado en un recorrido de `mac_online` y en uno del lector M1, antes y después |
| [`mutations-cpu.json`](mutations-cpu.json), [`mutations-cuda.json`](mutations-cuda.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`parity_capture.py`](parity_capture.py), [`parity_compare.py`](parity_compare.py) | Captura y comparación bit a bit |
| [`work_counts.py`](work_counts.py) | Cuentas de SHA-256, JSON, sincronizaciones y lanzamientos |
| [`mutations.py`](mutations.py) | Mutaciones, cada una con sus pruebas |

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

Los commits comparados son `develop` en 7a82b055 y la rama en 6a013588. Antes de integrar #565 la misma comparación, entre b9b1f36d y la rama sobre él, dio el mismo resultado. Esta igualdad está comprobada en el corpus técnico. La comparación con los eventos reales de la campaña queda pendiente con las medidas de tiempo, por el motivo que se explica más abajo.

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

## Medidas de tiempo pendientes

El cuarto criterio de #549, que el recorrido por evento de `mac_online` y de los lectores con banco mejore de forma medida, sigue abierto. Faltan las rondas de tiempo y VRAM, los perfiles de py-spy y Nsight Systems y el benchmark del CUDA Graph de la emisión. Hay dos motivos.

- **GPU con una carga ajena.** Un `llama-server` de otro proyecto del usuario ocupaba unos 5,4 GB y un 84 % de uso a las 14:36, y a las 18:34 seguía con 5.438 MiB y un 91 %. Las medidas de tiempo y VRAM solo se toman con la GPU libre de cargas ajenas, así que no se publica ninguna cifra tomada con esa carga.
- **Eventos sin vista.** Los eventos guardados de la vista `US+CN/fold-012` de #363 ya no se pueden abrir, porque la ejecución de la v3.1 sustituyó las muestras de la vista v3 activo a activo (la misma dependencia que describe el [informe del cableado de núcleos](../campaign-kernels-wiring-20261010/README.md#dependencias)). Hacen falta una vista v3.1 del mismo fold y eventos nuevos guardados desde ella.

Las órdenes para repetirlas, desde la raíz de cada árbol (`develop` y la rama) y con `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src`:

1. Eventos, una vez y sin GPU: `python benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`.
2. Rondas alternas `develop`, rama, rama, `develop`, cada una en su propio trabajo `memslot gpu`: `python benchmarks/chronological_replay.py VISTA/manifest.json TRABAJO --family titans --arm titans_mac_online --strict --warmup 2 --segments 8 --events eventos.pkl --capture CAPTURA.pt --output resultados.jsonl --label ETIQUETA`. Para los lectores, `--family readout --arm mars_titan_m1` y `--arm mars_titan_m3` con `--option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728`.
3. Bits de las rondas: `python reports/engineering/profiling-20261010/compare_capture.py DIRECTORIO salida.json develop:rama`.
4. Perfiles: `py-spy record --rate 100 --format raw` y `MARS_TITAN_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` sobre la orden de la ronda, resumidos con `benchmarks/collapsed_profile_summary.py` y `benchmarks/nsys_range_summary.py`.
5. Emisión con CUDA Graph: `python benchmarks/titans_emission_graph.py eventos.pkl VISTA/manifest.json salida.json 1024`.

Solo con esas medidas se decidirá si compensa capturar CUDA Graphs por forma exacta o fusionar la actualización de memoria en un núcleo CUDA, que este informe no evalúa.

## Reproducción

- Pruebas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/models/titans tests/training/test_financial_run.py tests/training/test_mars_titan_run.py tests/test_learning_hold_lowering.py tests/training/test_learning_hold_inventory.py`. Con `cuda:0` visible pasaron 771 pruebas en 8 min 20 s, con 2,1 GiB de memoria residente máxima. Antes de integrar #565 pasaron también los archivos de campaña y validación por ventanas de Titans-MAC y MARS-TITAN (157 pruebas y una omitida). `tests/training/test_titans_campaign.py` necesita por sí solo 5,4 GiB de memoria residente en `develop`, así que conviene ejecutarlo en un proceso aparte si el límite es de 6 GiB.
- Igualdad de bits: `python parity_capture.py ÁRBOL salida.pt DISPOSITIVO DTYPE TRABAJO` en cada árbol, con el mismo `TRABAJO` en los dos porque las huellas del corpus incluyen su ruta, y después `python parity_compare.py develop.pt rama.pt`.
- Trabajo contado: `python work_counts.py ÁRBOL salida.json TRABAJO`.
- Mutaciones: `python mutations.py ÁRBOL salida.json cpu` y `... cuda`.

Todas las órdenes usan `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dos hilos y `PYTHONPATH=ÁRBOL/src`, y las del lector el enlace episódico de `MARS_TITAN_EPISODIC_NATIVE`. Las pruebas, capturas y recuentos se ejecutaron en una RTX 4070 Laptop con torch 2.14.0 y CUDA 13.0.
