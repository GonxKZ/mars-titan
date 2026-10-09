# Entrenador cronológico de Titans-MAC

`training/financial_run.py` ajusta los cuatro controles de [`FinancialPredictor`](titans-financial-adapter.md) (`transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`) recorriendo los instantes de decisión en orden. Está implementado y comprobado técnicamente en CPU sin pasos de optimizador. No se ha ejecutado ningún entrenamiento: el bloqueo de #171 sigue vigente y el test de 2024 permanece cerrado.

## Recorrido y orden de cada evento

El índice de [observaciones financieras](financial-session-v2.md) agrupa por instante las entradas, las maduraciones y el cierre administrativo de una fase. El entrenador recibe cada instante en bloques de hasta `block_rows` activos, en el orden canónico del índice (mercado y símbolo). Cada bloque usa el estado previo de sus flujos y las mismas reglas de validación de entrada que el consumidor congelado.

Dentro de un evento el orden es fijo:

1. Se resuelven las etiquetas que maduran en ese instante contra la predicción realmente emitida y se acumula su pérdida en el tramo.
2. Si el tramo ya contiene `truncation` instantes de decisión, se calcula la pérdida media, se ejecuta un paso, se vuelven a sellar los parámetros y se cortan los grafos de todos los flujos.
3. Se predicen las entradas del instante con los parámetros vigentes.

Una etiqueta solo entra en la pérdida desde su evento de maduración y nunca antes. Su efecto llega en la actualización del primer límite de tramo posterior o igual a esa maduración, que precede a las predicciones de ese instante. Este orden difiere del banco episódico, que predice primero y aplica después los resultados del mismo instante. Sigue siendo causal porque `target_available_at` no supera el corte, igual que la disponibilidad de las entradas. La partición ya excluye las etiquetas que maduran en el límite de la validación, de modo que ningún ajuste usa información del tramo evaluado.

Las fases son tres. El calentamiento observa entradas sin grafo ni predicciones emitidas. El ajuste emite predicciones con grafo y actualiza en los límites de tramo. La validación usa parámetros congelados, reinicia la memoria rápida, repite su propio calentamiento y puntúa las predicciones emitidas con `SessionErrors.session_mae`, la misma métrica que usan las referencias.

## BPTT truncado

El tramo diferenciable cuenta instantes de decisión. Cada flujo conserva el grafo de sus observaciones dentro del tramo, como máximo `truncation`, y el paso del optimizador obliga a cortarlo, porque modifica los parámetros que guardó ese grafo. Por eso el BPTT es por activo dentro del tramo y la primera predicción de cada tramo solo retrocede un paso.

Si una etiqueta madura después del límite de su tramo, su grafo ya no existe y la predicción no aporta gradiente. El contador `labels_without_graph` lo registra. Con etiquetas de la sesión siguiente y un único calendario de mercado vale cero, como comprueban las pruebas. Una fase conjunta de Estados Unidos y China puede alternar instantes de ambos mercados y necesita medir ese contador antes de fijar `truncation`.

Los pesos rápidos iniciales M0 son parámetros aprendidos. En `mac_online` solo reciben gradiente en los tramos que contienen el primer estado de un flujo. En `mac_frozen` no hay escrituras y la memoria de cada flujo es M0, así que cada tramo vuelve a leer el M0 vigente con grafo. Sin esa regla, los flujos leerían copias antiguas de M0 tras cada paso y el ajuste no coincidiría con la inferencia congelada.

## Estados y parámetros

| Estado | Contenido | Quién lo cambia | Persistencia |
| --- | --- | --- | --- |
| Parámetros compartidos | Codificadores, fusión, cabeza, atención, proyecciones de MAC y de la memoria | Optimizador, rol `shared` | Modelo del checkpoint |
| Memoria persistente | `mac.persistent` | Optimizador, rol `persistent_memory`. Fija al validar | Modelo del checkpoint |
| Pesos rápidos iniciales | `mac.memory.initial_weights` | Optimizador, rol `initial_fast_weights` | Modelo del checkpoint |
| Pesos rápidos y momentum por flujo | `FinancialState.mac` | Regla asociativa de Titans al observar entradas. Nunca el optimizador | Bloques CPU de `export_state_cpu` |
| Cursor y cola | Último ID y corte por flujo, predicciones emitidas pendientes | Recorrido | Checkpoint de recuperación |
| Estado de trabajo | Grafo y representación de una predicción | Se descarta al cerrar el tramo | No se guarda |

`parameter_roles` construye esa partición y el entrenador exige que el optimizador cubra exactamente esos parámetros. Las pruebas comprueban que los tensores del estado rápido no son parámetros y que cada control recibe gradiente solo donde lo usa. `mac_disabled` no lo propaga a la memoria, la consulta ni la memoria persistente. `mac_frozen` no lo propaga a las proyecciones de escritura.

## Pérdida, selección y presupuesto

La pérdida principal es el MAE residual. La receta admite también MSE y Huber con el mismo validador que las referencias. Cada paso promedia las etiquetas maduras del tramo.

La selección reutiliza `selection.py`. El estado inicial se evalúa antes de ajustar y es elegible como mejor estado. La paciencia cuenta validaciones completas con la mejora mínima declarada. Con `stopping="fixed_budget"`, la regla común de `selection.py` recorre todas las épocas, conserva el mejor estado y registra en `plateau_epoch` dónde habría parado la paciencia, para que los controles emparejados tengan el mismo número de actualizaciones. Con los mismos eventos, los cuatro controles hacen los mismos pasos, cortes y etiquetas en la pérdida.

La receta propuesta está en [`configs/titans/chronological-training.json`](../../configs/titans/chronological-training.json), marcada como no ejecutada. Ventanas, calentamiento y épocas dependen del protocolo de #363.

## Checkpoints y reanudación

`save_training_state` conserva los dos estados de recuperación más recientes y el mejor estado aparte. Un estado de recuperación incluye modelo, optimizador, RNG, cursor, selección, historial, pesos rápidos y momentum por flujo, cola pendiente, errores acumulados y contadores. La barrera se sitúa justo después de una actualización, cuando no queda ningún grafo vivo. El cursor `stage="inputs"` indica que las etiquetas y el paso de ese evento ya están aplicados y que falta predecir sus entradas.

Se guarda cada `checkpoint_updates` pasos, cada `checkpoint_seconds` y ante una solicitud de parada. La validación no se reanuda a mitad, porque es determinista y se repite desde su inicio. El mejor estado se guarda tras validar, sin estado rápido, ya que cada recorrido reinicia la memoria. Al terminar, el predictor queda con los parámetros seleccionados.

Con un optimizador real de PyTorch, `run` se niega a empezar mientras la protección local de aprendizaje declare `training_allowed=false`.

## Identidades

La identidad de cada ejecución reúne la receta, la configuración del predictor y su variante, la huella de los parámetros iniciales, el recibo de `copy_paired_parameters` cuando existe, los roles de parámetros, el tipo de optimizador, las huellas del corpus y de ambos índices con sus fases, el código de los módulos implicados y la configuración numérica. `run_id` es su SHA-256. Cada variante obtiene una identidad propia aunque parta de los mismos parámetros copiados. La reanudación rechaza cualquier cambio de esa identidad.

Los parámetros cambian únicamente en el paso del optimizador. Después del paso se vuelve a sellar su huella y los estados de flujo, ya desconectados, se asocian a esa huella. `financial.py` no se ha modificado, así que las identidades del consumidor congelado y de sus recibos se conservan.

## Lector por bloques

El lector anterior decodificaba un grupo Parquet completo por cada observación. `batched_events` conserva el último grupo decodificado de cada activo, lee una vez por recorrido las etiquetas y los precios de cada activo y monta bloques por instante con la misma selección de filas. El índice vuelve a confirmar todos los archivos al terminar. La caché descarta grupos por antigüedad de uso con un presupuesto de 1 GiB por defecto.

| Medida (CPU, dos hilos, carga media cercana a 11 por la codificación en curso) | Por observación | Por bloques | Factor |
| --- | --- | --- | --- |
| Corpus técnico de 16 activos, grupos de 128 filas y anchuras reales, fase de ajuste | 5.520 µs | 212 µs | 26,1 |
| Mismo corpus, fase de validación | 5.144 µs | 199 µs | 25,8 |
| Archivos v3 reales (AA, AFMC y AROW), solo lectura | 3,4 a 3,5 ms | 61 a 112 µs | Estimado |

En los archivos reales se mide la decodificación de grupos de 128 filas y se reparte entre sus filas. No es un recorrido completo, porque esa edición todavía no tiene objetivos. Ambos lectores producen la misma huella de recorrido. El benchmark está en [`benchmarks/financial_observations.py`](../../benchmarks/financial_observations.py).

Un perfil de una pasada de ajuste en CPU, 16 activos, anchura 64 y `truncation=8`, sin pasos de optimizador, tarda unos 1,7 ms por observación en `mac_online`. La preparación del predictor ocupa un 50 % y el backward un 28 %. Dentro de la preparación, las comprobaciones de finitud del núcleo suman cerca de 0,6 s de 3,4 s. En CUDA cada una sincroniza el dispositivo, así que su coste debe medirse antes de entrenar. El lector queda en un 13 %. En `transformer_direct` la pasada tarda 1,0 ms por observación y el lector pesa un 24 %.

## Correspondencia con `CorpusDataset`

El índice puntúa una etiqueta de la partición si su decisión pertenece a `[decision_start, decision_end)` y madura antes de `decision_end`. `CorpusDataset.batches` entrega todas las etiquetas aceptadas de la partición. Las pruebas comprueban que ambos conjuntos coinciden cuando la fase cubre la partición completa. Si el calentamiento ocupa parte de la partición, las etiquetas de ese tramo se observan sin puntuarse. Si la fase termina antes que la partición, también se excluyen las etiquetas que maduran en su límite. La diferencia coincide exactamente con esas filas.

## Comprobaciones

Las pruebas usan un optimizador propio que registra gradientes y llamadas sin heredar de `torch.optim.Optimizer` ni modificar pesos. Comprueban:

- invariancia de predicciones y gradientes pasados al alterar entradas, precios y etiquetas posteriores a un corte,
- orden canónico, bloques idénticos al lector por observación y límites de tramo exactos,
- que ninguna etiqueta entra en la pérdida antes de madurar y que cada una se usa una vez,
- reanudación tras un corte con la misma secuencia de gradientes, predicciones e historial que la ejecución continua,
- gradientes por rol según el control,
- predicciones de validación iguales al consumidor congelado sobre el lector por fila y, con el enlace nativo CPU, iguales a `FinancialSession`,
- correspondencia de filas puntuadas con `CorpusDataset`,
- llamadas al optimizador, `zero_grad` y sellado de parámetros exactamente en cada actualización,
- selección con secuencias de métricas fijadas, presupuesto fijo y estado inicial elegible.

Se aplicaron diecinueve mutaciones dirigidas sobre el entrenador y el lector y las pruebas detectaron dieciocho. La restante era equivalente, porque retiraba una condición redundante, y el código se simplificó. El [recibo](../../reports/engineering/titans-chronological-trainer-20261009.json) conserva las medidas, pruebas y mutaciones.

## Observación técnica sobre `mac_online`

Con los parámetros iniciales, `α` vale cerca de 0,5 porque las tasas no tienen bias. La memoria rápida se reduce a la mitad en cada observación. En el fixture, la norma de cada capa pasa de 5,6 a 0,09 tras seis observaciones y a 1,4·10⁻¹⁰ tras 36. La representación previa a la cabeza cae por debajo de 10⁻¹⁵ y la predicción queda en el bias de la cabeza. El gradiente de la cabeza baja de 10⁻⁷ a 10⁻¹³ en cuatro tramos. `mac_frozen` mantiene M0 y no presenta ese descenso.

Es una propiedad del núcleo sin entrenar, no un resultado predictivo. Puede impedir que `mac_online` aprenda la escritura. Cambiar la inicialización o añadir bias a las tasas exige una identidad nueva del núcleo y su propio contraste, por lo que no se ha modificado aquí.

## Pendiente

- Ejecución real tras levantar el bloqueo, con el protocolo y las ventanas de #363.
- Comprobación CUDA escrita y sin ejecutar: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 UV_PROJECT_ENVIRONMENT=<entorno> uv run --no-sync python -m pytest -q tests/training/test_financial_run.py::test_cuda_pass_matches_cpu_without_optimizer_steps`. Después hay que perfilar una pasada en `cuda:0` con FP32, medir memoria y sincronizaciones y comprobar la recuperación en ese dispositivo.
- Banco episódico, M1 a M3, K mayor que 1 y C siguen fuera de este entrenador, porque `FinancialPredictor` exige banco desactivado y K=1 y el factorial CM-v1 tiene su propio contraste.
- Decidir la inicialización de las tasas de `mac_online` antes de fijar la receta.
- Medir `labels_without_graph` en fases conjuntas de dos mercados.
