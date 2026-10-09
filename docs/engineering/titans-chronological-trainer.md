# Entrenador cronológico de Titans-MAC

`training/financial_run.py` ajusta los cuatro controles de [`FinancialPredictor`](titans-financial-adapter.md) (`transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`) recorriendo los instantes de decisión en orden. Está implementado y comprobado técnicamente en CPU sin pasos de optimizador. [`titans_walk_forward.py`](#ventana-walk-forward) lo usa para ajustar una ventana del protocolo desde 2000 y escribir predicciones por fila. No se ha ejecutado ningún entrenamiento: el bloqueo de #171 sigue vigente y el test de 2024 permanece cerrado.

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

Se guarda cada `checkpoint_updates` pasos, cada `checkpoint_seconds` y ante una solicitud de parada. La validación no se reanuda a mitad, porque es determinista y se repite desde su inicio. El mejor estado se guarda tras validar, sin estado rápido, ya que cada recorrido reinicia la memoria. Al terminar, el predictor queda con los parámetros seleccionados, cargados desde el mejor checkpoint con la huella que registra el informe. Reanudar una ejecución ya completa también carga ese estado. Antes se quedaba con el último checkpoint de recuperación, que con presupuesto fijo puede ser posterior al mejor.

Con un optimizador de `torch.optim`, `run` llama a `require_learning_allowed` antes de abrir la salida y lanza `LearningHoldError` mientras la protección local declare `training_allowed=false`. Las pruebas recorren el bucle con un optimizador propio que solo registra gradientes.

## Inferencia de tramos posteriores

`ChronologicalInference` reúne el recorrido por eventos que comparten el ajuste y la inferencia congelada, y `ChronologicalTrainer` lo amplía con el grafo, la pérdida y el paso. `predict` recorre validación, calibración o evaluación con la misma regla que la validación del ajuste: parámetros congelados, memoria rápida reiniciada, calentamiento de la fase sin etiquetas y predicciones emitidas antes de conocer su resultado. Exige un tramo con la misma entrada. `predict_partition` del entrenador exige además que sea del mismo corpus y posterior al ajuste. Cada etiqueta resuelta entrega a un destino de filas el flujo, la decisión, la predicción, el objetivo y, con la cabeza de cuantiles, los cinco niveles emitidos.

## Memoria del tramo y acumulación por bloques

Hasta la actualización, el recorrido conserva el grafo de todas las predicciones del tramo y del estado de todos sus flujos. La memoria viva crece con los activos por instante multiplicados por `truncation`. [`benchmarks/titans_chronological_memory.py`](../../benchmarks/titans_chronological_memory.py) lo mide con la receta de cuantiles (FP32, anchura 64, `truncation=8`) y las dimensiones reales de la edición histórica, en CPU y con entradas aleatorias. Cuenta los almacenamientos únicos guardados para el backward y alcanzables desde las salidas y el estado del tramo, sin parámetros.

| Control | Grafo guardado por fila e instante |
| --- | --- |
| `transformer_direct` | 225 KB |
| `mac_disabled` | 226 KB |
| `mac_frozen` | 273 KB |
| `mac_online` | 375 KB |

El valor por fila es el mismo con 64, 128 y 256 filas. La mayor parte corresponde al codificador de precios sobre 64 sesiones. Con `mac_online`, el grafo vivo antes del backward sería:

| Activos por instante | Por evento | Tramo de 8 instantes | Con `accumulation_rows=128` |
| --- | --- | --- | --- |
| 128 | 48 MB | 0,38 GB | 0,41 GB |
| 1.024 | 384 MB | 3,1 GB | 0,57 GB |
| 4.202 (todo EE. UU.) | 1,6 GB | 12,6 GB | 1,2 GB |
| 5.023 (todos los activos con precios) | 1,9 GB | 15,0 GB | 1,3 GB |

Son estimaciones lineales a partir de la medida, en unidades decimales, sin temporales de cada operación, contexto CUDA ni caché del asignador. Con más de unos mil activos por instante el tramo completo no cabe en los 8 GB de la GPU. El [recibo](../../reports/engineering/titans-chronological-memory-20261009.json) conserva las medidas de los cuatro controles.

`ChronologicalRecipe.accumulation_rows` es opcional y vale `None` por defecto, con la identidad de receta anterior intacta. Con un entero, cada bloque del evento se predice con el mismo cálculo que en el recorrido por defecto y su grafo se corta en cuanto se emite la predicción. El tramo guarda los lotes de entrada y el estado de cada flujo al empezar. Al actualizar, `_replay` recorre los flujos con etiquetas en bloques de hasta `accumulation_rows`. Cada bloque parte de ese estado inicial, repite los lotes en el orden original, de modo que la memoria rápida avanza en el mismo orden, y retropropaga su pérdida media multiplicada por su fracción de etiquetas. La suma de los bloques es la pérdida media del tramo, así que hay un único paso por actualización. Los flujos son independientes dados los parámetros, que no cambian dentro del tramo.

La memoria del tramo pasa a ser el grafo de un bloque, más los lotes de entrada (6,7 KB por fila e instante) y dos estados rápidos por flujo (64 KB cada uno en los controles con MAC). El coste previsto es un forward más por tramo. No se ha medido sobre el recorrido completo.

Las pruebas de [`test_financial_run_accumulation.py`](../../tests/training/test_financial_run_accumulation.py) comparan, en los cuatro controles, con cabeza escalar y de cuantiles y bloques de uno y dos flujos, un ajuste de dos épocas con y sin acumulación. Predicciones, etiquetas, actualizaciones, métricas de validación y llamadas al optimizador coinciden bit a bit y los gradientes de cada paso coinciden con tolerancia relativa 10⁻¹⁰ en FP64. También comprueban que la repetición no supera el bloque declarado, que el grafo recorrido por cada backward baja a un tercio con bloques de un flujo sobre los tres del fixture, que la reanudación reproduce la ejecución continua y que solo se exporta un recorrido en la barrera posterior a un paso.

Las recetas no declaran `accumulation_rows`. El valor para la campaña depende de los activos por instante y de la memoria medida en `cuda:0`.

## Ventana walk-forward

`run_titans_window` en [`training/titans_walk_forward.py`](../../src/mars_titan/training/titans_walk_forward.py) y el script [`run_titans_walk_forward.py`](../../scripts/run_titans_walk_forward.py) ajustan una variante en una ventana del protocolo v2 y escriben sus predicciones. Reciben la vista temporal con máscaras, el protocolo, la ventana, la receta, la variante, la semilla, la salida y el dispositivo, `cpu` o `cuda:0` explícito.

El orden es fijo:

1. `require_learning_allowed` se comprueba antes de leer ninguna fuente.
2. El protocolo se recibe como ruta o como documento, y su huella es la de su JSON canónico. Debe contener la ventana y la semilla. La receta debe declarar `epochs` igual a `max_epochs` y la misma selección que `stopping_rule(protocol)`, hoy presupuesto fijo de 30 épocas, mejor estado y `min_delta = 1e-5`.
3. La petición reúne las huellas de vista, protocolo y receta, la variante, la semilla, el dispositivo y el código. Si la salida ya contiene esa ventana completa, se comprueban las huellas de sus predicciones y se devuelve sin abrir la vista. Otra petición sobre la misma salida se rechaza.
4. En `cuda:0` se exige CUDA disponible y `CUBLAS_WORKSPACE_CONFIG` antes de construir el modelo.
5. Los contratos temporales de la vista deben coincidir con el protocolo y la ventana, con filas en los cuatro tramos y la reserva final cerrada.
6. El ajuste usa `ChronologicalTrainer` con parámetros iniciales copiados de `pairing_source`. Después se carga el mejor estado y `predict_partition` escribe `validation-predictions.parquet`, `calibration-predictions.parquet` y `evaluation-predictions.parquet` con el esquema común (`sample_id`, `asset_id`, `market`, `prediction_at`, `target`, `prediction` y `zero`) y los cinco cuantiles cuando la receta los declara. Cada etiqueta puntuada debe llegar al destino, y `check_view_rows`, la misma comprobación que usa la GRU candidata, exige exactamente las filas de la vista por mercado, activo e instante, con el mismo objetivo. Si no coinciden, informa de cuántas faltan, sobran, se repiten o cambian de objetivo. En validación, el MAE por sesión recalculado debe reproducir la puntuación seleccionada.
7. `run.json` guarda la petición, la identidad con la receta completa, el resumen del ajuste, el estado elegido en `checkpoint` y la huella, filas, bytes y métricas de cada tramo. Cada Parquet se escribe solo después de conciliar sus filas, y `run.json` se actualiza tras cada tramo confirmado.

Una pausa durante el ajuste se reanuda desde su último checkpoint coherente. Una pausa entre tramos conserva los ya escritos y continúa con el siguiente.

### Política de memoria en inferencia

La política `reset_each_pass_then_input_warmup_v1` es la misma para los cuatro controles:

- Cada recorrido, sea una época de ajuste o uno de los tres tramos predichos, empieza con el estado inicial de cada flujo.
- Antes de validación, calibración y evaluación se observan las entradas de los `warmup_months` anteriores, sin emitir predicciones ni puntuar etiquetas. El calentamiento no empieza antes del origen del ajuste. Las recetas declaran 12 meses.
- La memoria rápida solo se escribe en `mac_online`, con la regla asociativa sobre entradas. `mac_frozen` lee M0, `mac_disabled` no lee memoria y `transformer_direct` no tiene estado.
- Las etiquetas maduras se comparan con la predicción emitida y nunca se escriben en la memoria. El banco episódico sigue desactivado.

Reiniciar en cada tramo hace que la predicción de evaluación no dependa de haber recorrido antes la calibración y que los cuatro controles vean exactamente las mismas entradas. Encadenar la memoria entre tramos también sería causal, pero mezclaría la longitud de la historia con la variante. El valor de 12 meses es una declaración previa, no un ajuste con datos. Las pruebas alteran entradas posteriores al corte de cada tramo y anteriores al calentamiento, y comprueban que ninguna predicción de otro tramo cambia.

### Conexión con la campaña con máscaras

`training/masked_campaign.py` registra los ejecutores `("titans_mac", "fit")` y `("titans_mac", "carry")` junto a los de la GRU candidata. Solo crean trabajos si la configuración de campaña declara la sección opcional `titans_mac`, con el mismo patrón que `episodic_gru`:

```json
"titans_mac": {
  "recipe": "../titans/chronological-training-quantile.json",
  "arms": {"titans_mac_online": "mac_online", "titans_mac_frozen": "mac_frozen"},
  "search_seed": 42
}
```

`campaign_plan` valida la receta sin importar PyTorch: política de entradas con máscaras, nombre, cabeza `quantile_head_v1`, pérdida pinball, épocas y selección iguales a `stopping_rule(protocol)` y sección `walk_forward`. Cada brazo Titans de la comparación necesita una variante distinta. El caso de cada trabajo contiene la ruta y la huella de la receta, la variante y la semilla. Al no haber búsqueda de hiperparámetros, la semilla de búsqueda tiene un único candidato y las demás semillas son finalistas con el mismo caso. La huella de la receta forma parte de la identidad de cada trabajo. Sin la sección, el plan y la identidad de la campaña no cambian y `check` sigue listando la familia como pendiente.

- `titans_fit` recibe el `JobRun` del trabajo, comprueba que la receta conserva la huella planificada, toma el protocolo del contrato temporal de la vista y llama a `run_titans_window` sobre el intento del trabajo. Una pausa se traduce en la pausa de la campaña y el siguiente intento reanuda desde el checkpoint coherente. La campaña lee del informe las predicciones de calibración y evaluación, el MAE de validación y el estado elegido, y comprueba las mismas filas que los demás brazos.
- `titans_carry` predice una ventana trasladada de la variante B con `carry_titans`, que también exige el permiso de aprendizaje antes de leer nada. Comprueba que el ancla es una ventana de Titans completada sobre su vista, que dejó de aprender antes de la calibración trasladada (`carried_window`) y que el estado elegido conserva su huella. Carga ese estado en un predictor construido con la entrada de la nueva vista, y solo admite que cambien las huellas de la vista y del índice. Después predice calibración y evaluación con `ChronologicalInference`, sin ningún ajuste, aplica la misma comprobación de filas que la ventana y escribe `carry.json` con la huella del estado del ancla.
- La política de memoria de un traslado es la misma de la ventana, `carried_memory_policy`: los parámetros son los elegidos en el ancla, la memoria rápida nunca se transfiere desde el ancla y cada tramo empieza en el estado inicial con su propio calentamiento de 12 meses de entradas.
- El recibo de ventana de cada mercado lo escribe la campaña con el estado elegido como `parent`. En una ventana trasladada es el del ancla.

[`test_titans_campaign.py`](../../tests/training/test_titans_campaign.py) ejecuta en CPU una campaña B sobre US con las vistas v2 del corpus técnico. `titans_mac_online` usa los ejecutores reales con un optimizador que solo registra gradientes y los demás brazos son los dobles de la campaña. Se comprueban los 7 ajustes y 12 traslados del plan, las mismas filas y objetivos que la GRU de referencia en cada ventana, los cuantiles y la ausencia de filas de 2024, que cada traslado parte del estado elegido en su ancla con la política declarada, los 19 recibos de ventana, el manifiesto de fuentes y la repetición bit a bit de un traslado. También se rechazan anclas incoherentes, recetas cambiadas después de planificar, estados de otra arquitectura, predicciones sin cuantiles y traslados con una fila repetida o un objetivo cambiado.

### Comprobaciones de la ventana

[`test_titans_walk_forward.py`](../../tests/training/test_titans_walk_forward.py) usa un protocolo técnico con una ventana y el corpus cronológico con máscaras. Comprueba las fases de todas las ventanas US, CN y conjuntas con calentamiento 0, 12 y 60 meses (este último recortado en el origen), que los tres Parquet de los cuatro controles coinciden en filas y objetivos con `CorpusDataset.batches` y que `walk_forward_comparison` los acepta en una evaluación completa, que no hay filas de 2024, la copia de parámetros emparejados, la invariancia ante perturbaciones futuras y anteriores al calentamiento, la reanudación con gradientes y salidas idénticos, el rechazo de peticiones o archivos alterados, el bloqueo antes de crear nada, el rechazo de recetas y protocolos incoherentes, de vistas con la reserva abierta o un tramo vacío y de salidas con archivos ajenos, y que una fila perdida o un objetivo cambiado detienen la ventana sin dejar Parquet sin confirmar.

El [recibo de verificación](../../reports/engineering/titans-walk-forward-verification-20261009.json) registra cada archivo de pruebas ejecutado en CPU sobre develop `6119a7ea`, con 857 pruebas superadas y 4 omitidas que exigen `cuda:0` o vistas reales, y tres rondas de mutaciones dirigidas con 52 de 52 detectadas tras añadir las pruebas que motivaron las supervivientes. Ninguna ejecución dio pasos de optimizador.

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

Es una propiedad del núcleo sin entrenar, no un resultado predictivo. Puede impedir que `mac_online` aprenda la escritura. Cambiar la inicialización o añadir bias a las tasas exige una identidad nueva del núcleo y su propio contraste, por lo que no se ha modificado aquí. Esa identidad existe ahora con [bias declarado](titans-gate-initialization.md). `FinancialConfig` la expone como `gate_bias` y las recetas escalar y de cuantiles la declaran con semivida de 256 observaciones. Las medidas de este apartado corresponden a v1.

## Pendiente

- Ejecución real tras levantar el bloqueo, con el protocolo y las ventanas de #363.
- Comprobación CUDA escrita y sin ejecutar: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 UV_PROJECT_ENVIRONMENT=<entorno> uv run --no-sync python -m pytest -q tests/training/test_financial_run.py::test_cuda_pass_matches_cpu_without_optimizer_steps`. Después hay que perfilar una pasada en `cuda:0` con FP32, medir memoria y sincronizaciones y comprobar la recuperación en ese dispositivo.
- La ruta de la ventana no tiene todavía una comprobación CUDA propia. Con el bloqueo levantado, una ventana se lanzaría con `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src:. uv run --no-sync python scripts/run_titans_walk_forward.py --view <vista>/manifest.json --protocol <protocolo> --window <ventana> --recipe configs/titans/chronological-training-quantile.json --variant mac_online --seed 42 --output <salida> --device cuda:0`.
- Medir en `cuda:0` el pico del asignador de un tramo antes de fijar `accumulation_rows`: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONPATH=src:. uv run --no-sync python benchmarks/titans_chronological_memory.py --device cuda:0 --output reports/engineering/titans-chronological-memory-cuda-<fecha>.json`. Con `--rows` igual al bloque elegido se obtiene el pico de la repetición. El coste del forward repetido sobre el recorrido completo tampoco se ha medido.
- Banco episódico, M1 a M3, K mayor que 1 y C siguen fuera de este entrenador, porque `FinancialPredictor` exige banco desactivado y K=1 y el factorial CM-v1 tiene su propio contraste.
- Repetir con `gate_bias` la observación de normas y gradientes del fixture de este entrenador. La inicialización ya está declarada en las recetas, pero esa medida concreta se hizo con v1.
- Medir `labels_without_graph` en fases conjuntas de dos mercados.
