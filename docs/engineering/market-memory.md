# Memoria de mercado compartida (PT4)

Estado a 10 de octubre de 2026: **implementada y comprobada sin entrenar**. El resultado experimental está pendiente, porque el bloqueo de aprendizaje impide el piloto y la comparación. Tarea [#456](https://github.com/GonxKZ/mars-titan/issues/456). La motivación, el control obligatorio y la predicción están en las [propuestas posteriores a Titans](../research/post-titans-proposals.md#pt4-memoria-de-mercado-compartida-línea-posterior). Es una línea posterior y no entra en la campaña A.

## Qué hace

El núcleo trata cada activo como un flujo independiente: lo que ocurre en un activo no llega a otro salvo por el banco episódico. PT4 añade un contexto común a los flujos de un mercado, con dos formas que se declaran con `FinancialConfig.session_context` y solo con `mac_online`:

- `market_memory_v1`, la propuesta: una segunda memoria neuronal de Titans por mercado (US y CN), de dimensión H y profundidad 2, con el sesgo de puertas y la normalización residual del núcleo y una semilla propia. Cada flujo la lee con su propia consulta. Se escribe una sola vez por sesión, después de emitir todas sus predicciones, con la media de los gradientes internos de los tokens de esa sesión.
- `sector_mean_v1`, el control obligatorio de la revisión adversarial: cada flujo recibe la media de los tokens fusionados de su sector en la última sesión con miembros, sin memoria ni olvido aprendido. El reparto sale de las fichas de FinMultiTime (`SectorMap`).

El contexto entra en el token antes de MAC: x̃_t = x_t + W_o c_t, con W_o inicializada a cero. Con la misma semilla el predictor arranca exactamente igual que su control, porque la memoria y la consulta usan generadores propios y no desplazan el generador global. MAC ve x̃_t en la consulta, la atención y la escritura de la memoria de cada flujo. Lo que se escribe en el contexto y lo que devuelve `detached_tokens` es x_t, el token fusionado sin contexto, para que el contexto no se alimente de sí mismo.

El código está en `session_context.py` (`SessionContext` y su estado) y `neural_memory.py` (`read_shared`, `update_shared` y la escritura común `_write`) de `src/mars_titan/models/titans/`, en `src/mars_titan/data/sector_map.py` y en `financial.py`, que añade `prepare(..., context=...)` y los métodos `initial_context`, `commit_context`, `detach_context`, `export_context` y `restore_context`. El recorrido cronológico (`training/financial_run.py`) lo conecta. La sesión congelada y los demás consumidores no pasan contexto, así que un predictor con PT4 falla al preparar en ellos en lugar de ignorarlo.

## Ecuaciones

Para un mercado y una sesión s con n flujos f y tokens fusionados x_f:

**Lectura.** q_f = W_q x_f / ‖W_q x_f‖ y c_f = M_{s−1}(q_f), donde M_{s−1} es la memoria tras confirmar las sesiones anteriores a s. La memoria aplica la misma red x + LN(MLP(x)) que el núcleo.

**Escritura, tras emitir la sesión.** Con k_f = normalize(W_K x_f) y v_f = W_V x_f, la pérdida asociativa es la media de las n pérdidas en los mismos pesos:

$$\ell(W) = \frac{1}{n}\sum_f \lVert M_W(k_f) - v_f \rVert^2, \qquad \nabla = \frac{1}{n}\sum_f \nabla \ell_f.$$

Las puertas salen de la media de los logits. Como las proyecciones son lineales, coinciden con los logits del token medio: a = W_α x̄ + b_α, y lo mismo para η y θ. Después se aplica la regla de Titans sin cambios, S_s = η S_{s−1} − θ ∇ y W_s = (1 − α) W_{s−1} + S_s, una vez por sesión. Con una sola fila la escritura es bit a bit la de un token del núcleo.

**Control sectorial.** c_f = x̄_{g(f)}, la media de los tokens del grupo g(f) en su última sesión con miembros, y cero antes de la primera. Los grupos son (mercado, sector) y un grupo desconocido por mercado.

| Pieza | Fuente o derivación | Módulo | Pruebas |
| --- | --- | --- | --- |
| Memoria, puertas y regla de escritura | Titans (Behrouz et al., 2025), sin cambios | `NeuralMemory._write`, `update` | `test_a_shared_write_of_one_row_is_the_ordinary_write_of_one_token` |
| Lectura compartida con escritura por gradiente medio de la sesión | Derivación propia | `read_shared`, `update_shared`, `SessionContext._read` | `test_a_shared_write_takes_the_mean_inner_gradient_and_the_gates_of_the_mean_logits`, `test_the_market_read_uses_the_own_query_normalized_and_the_output_gain`, `test_the_shared_memory_rejects_per_flow_options_batches_and_other_shapes` |
| Media sectorial de la sesión anterior | Control declarado en la propuesta | `SessionContext._sector_means`, `SectorMap` | `test_the_sector_control_keeps_the_mean_token_of_each_sector_and_leaves_absent_ones`, `test_sector_map_reads_both_descriptions_and_never_invents_a_sector`, `test_sector_map_rejects_conflicts_missing_columns_and_links` |
| Lectura solo de sesiones confirmadas | Contrato de causalidad intrasesión | `SessionContext.read`, `commit` | `test_a_session_reads_only_sessions_already_committed`, `test_reads_reject_mixed_sessions_repeated_flows_and_reopened_sessions` |
| Independencia de la partición en bloques | Orden canónico de flujos antes de escribir | `SessionContext.commit` | `test_the_block_partition_changes_neither_reads_nor_the_committed_state`, `test_the_pass_does_not_depend_on_the_block_size` |
| Arranque en el control y entrada antes de MAC | Ganancia W_o en cero y semillas propias | `FinancialPredictor._prepare` | `test_the_predictor_starts_exactly_at_its_control_with_the_same_seed`, `test_the_context_enters_the_token_before_the_memory_and_needs_its_state` |
| Gradiente entre sesiones y corte en la barrera | BPTT truncado del recorrido | `commit(differentiable=True)`, `detach` | `test_a_later_read_differentiates_through_the_committed_write`, `test_a_commit_without_gradient_keeps_no_graph_of_the_session_tokens` |
| Recorrido, recuperación y truncamiento | `financial_run.py` | `_observe`, `_update`, `_export`, `_restore` | `test_the_validation_pass_commits_each_session_once_after_emitting_it`, `test_the_checkpoint_keeps_the_context_and_the_truncation_detaches_it`, `test_export_and_restore_continue_bit_for_bit`, `test_block_accumulation_is_rejected_with_pt4` |
| Declaraciones inválidas | Fail fast | `FinancialConfig.__post_init__` | `test_declarations_fail_before_building` |
| Comparación declarada sin ejecutar | Regla escrita de antemano | `configs/evaluation/market-memory-comparison.json` | `test_comparison_is_declared_with_both_controls_and_its_budget_but_not_executed` |

Las pruebas están en `tests/models/titans/test_market_memory.py`. Calculan lecturas, escrituras y gradientes con tokens técnicos y con el corpus cronológico de prueba, sin objetivos reales y sin ningún paso de optimizador. Los recorridos usan `evaluate` o llaman a la barrera sin etiquetas en la pérdida.

## Entradas, orden y estado

PT4 solo lee los tokens fusionados de las decisiones, igual que la memoria de Titans. No usa etiquetas ni errores. En el recorrido cada evento es un instante. Cada bloque del evento lee el contexto confirmado y deja sus tokens pendientes, y cuando se han emitido todos los bloques el contexto se confirma una vez por mercado. Una sesión no puede leer nada de su propia sesión, y `read` rechaza dos instantes del mismo mercado en un bloque, un flujo repetido, una sesión nueva con otra pendiente y una sesión ya confirmada. Antes de escribir, los tokens pendientes se ordenan por flujo, así que el resultado no depende de cómo se repartan en bloques.

El calentamiento también escribe el contexto, como escribe la memoria de cada flujo. El cierre de fase lo reinicia. En el ajuste el gradiente de una predicción llega, a través de la escritura confirmada, a los tokens de otros flujos de sesiones anteriores del mismo tramo y a W_q, W_o y las proyecciones de la memoria de mercado. La barrera de truncamiento lo corta con la nueva identidad de parámetros, igual que el estado de los flujos. El checkpoint guarda el contexto confirmado, porque entre eventos nunca queda nada pendiente.

El estado compartido no crece con el número de activos: una memoria de lote 1 por mercado (pesos y momentum de dos capas, 64 KiB por mercado en FP32 con H = 64) o una matriz de medias con una fila por grupo (44 con las fichas actuales).

## Control con el que se descarta

La [declaración](../../configs/evaluation/market-memory-comparison.json) fija tres modelos sobre la receta de 2000 con `mac_online`: el control sin contexto, la memoria de mercado y el control sectorial. Comparten filas, ventanas, semillas, pesos iniciales, casos de búsqueda, actualizaciones del optimizador, regla de parada, truncación, tamaño de bloque y la receta sin acumulación por bloques. La declaración registra el presupuesto declarado (8 h de GPU) frente al consumido (cero). La memoria de mercado se conserva solo si el límite superior del intervalo del 95 % de la diferencia relativa de MAE por sesión es negativo frente al control sectorial y frente al control sin contexto. Se abandona en otro caso o si alguna predicción depende de otra fila de su misma sesión. La predicción registrada es una mejora pequeña concentrada en sesiones con movimientos sectoriales, nula en promedio, y que el control sectorial la igualará.

## Coste medido sin entrenar

[`cost.json`](../../reports/engineering/market-memory-20261010/cost.json) mide en CPU, en FP32 y con dos hilos, los tres modelos de la declaración con el mismo script y las mismas entradas reales sin objetivos que PT5 y PT6: 128 flujos de EE. UU. y 8 instantes, un bloque por instante. Cada repetición recorre los instantes con grafo, lee el contexto, confirma cada sesión tras emitirla, calcula una pinball frente a ceros y ejecuta el backward, sin optimizador, y después repite el recorrido sin grafo. Las 15 repeticiones se alternan entre modelos con orden rotatorio. El control sectorial usa las fichas reales del conjunto (44 grupos y 5.071 flujos asignados).

| Modelo | Parámetros añadidos | Contexto compartido | Forward con grafo | Backward | Inferencia |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 0 | 0 | 0,252 s | 0,110 s | 0,241 s |
| Memoria de mercado | 28.866 | 131.072 B | 0,292 s (1,16) | 0,138 s (1,25) | 0,276 s (1,14) |
| Control sectorial | 4.096 | 11.264 B | 0,265 s (1,05) | 0,132 s (1,20) | 0,265 s (1,10) |

Son medianas, con el cociente frente al control entre paréntesis, y con otras cargas en la máquina (carga media cercana a 12). Una medida anterior con el mismo orden alternado dio 1,11, 1,19 y 1,16 para la memoria de mercado y 0,99, 1,05 y 1,06 para el control sectorial. El estado por flujo no cambia (65.552 B). El contexto compartido son las dos memorias de mercado de lote 1 o la matriz de 44 medias, y no crece con el número de activos.

La primera medida del control sectorial costaba un 49 % más que el control en inferencia. Revisando el código, la identidad del predictor se recalcula en cada decisión e incluía la del reparto, que serializaba los 5.071 flujos cada vez. `SectorMap` calcula ahora su identidad una vez al construirse y `SessionContext` guarda su huella. La identidad resultante es la misma y, con el cambio, el sobrecoste medido bajó a los valores de la tabla. No se ha perfilado con una herramienta. No se ha medido en `cuda:0`.

## Desviaciones y límites

- La repetición por bloques de `accumulation_rows` no reconstruye el contexto entre bloques, así que el recorrido rechaza PT4 con acumulación. La receta base la usa para que `mac_online` quepa con US y CN completos. El piloto de 64 activos cabe en un bloque por instante y los tres modelos se declaran sin acumulación. Una comparación a escala completa exige antes implementar esa repetición.
- La propuesta pedía todos los activos de una sesión en un solo lote. La implementación admite varios bloques por sesión porque acumula los tokens pendientes y escribe al final, pero cada sesión de un mercado debe completarse antes de abrir la siguiente.
- Las puertas α, η y θ son una por sesión y mercado, no por flujo, porque la memoria es única.
- La memoria de mercado tiene su propia configuración: no hereda PT1 ni PT5 aunque el núcleo los declare. Su estabilidad depende de las puertas de Titans, sin certificado.
- Los pesos iniciales de la memoria de mercado son parámetros, pero el calentamiento no es diferenciable y el contexto se crea en el primer evento de la fase. En la práctica conservan su inicialización, igual que los pesos iniciales de los flujos que aparecen durante el calentamiento.
- El reparto sectorial es el de las fichas actuales de FinMultiTime y no una serie histórica. Una empresa reclasificada después aparece con su sector posterior en todos los años. Los activos con `N/A`, `Client Error`, sin sector o ausentes de la ficha van al grupo desconocido de su mercado, nunca a un sector inventado. La identidad guarda la huella de las fichas y del reparto con `point_in_time=False`.
- PT4 no se combina con el control local C.
