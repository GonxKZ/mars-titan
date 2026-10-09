# Campaña de entrenamiento sobre la edición histórica desde 2000

Revisión del 9 de octubre de 2026. Este documento ordena el recorrido completo desde los datos hasta la comparación final. Describe un plan y el estado de cada etapa. No contiene resultados predictivos: ningún modelo se ha entrenado sobre esta edición y el test de 2024 sigue cerrado.

## Objetivo

Entrenar desde cero todas las arquitecturas comparadas con todos los datos utilizables desde 2000, con las mismas filas, la misma validación temporal y presupuestos comparables. Después se aplican postentrenamientos y políticas de refuerzo sobre esas mismas bases. La pregunta es cuánto cambia el error de cada variante frente a sus referencias, con incertidumbre temporal y costes medidos.

El bloqueo de aprendizaje sigue vigente hasta que la edición esté completa y verificada. Su condición es preparar la edición desde 2000 con ausencias y máscaras explícitas y la misma población para todos los modelos.

## Etapas y estado

| Etapa | Contenido | Estado a 9 de octubre | Tarea |
| --- | --- | --- | --- |
| 1. Edición de entradas | Codificación v3 de precios, noticias, gráficos, fundamentales y 140 posiciones macro con nivel, presencia y antigüedad, más cinco bits de presencia por modalidad | En curso. Más de 750 de unos 5.028 activos elegibles confirmados. El resto se codifica en paralelo con paridad bit a bit comprobada | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 2. Objetivos | Retorno residual apertura-cierre de la sesión siguiente, OLS con 252 sesiones y un mínimo de 126 pares, solo con pares disponibles en la decisión | Código existente. Exige la población completa, así que espera a la etapa 1 | [#15](https://github.com/GonxKZ/mars-titan/issues/15) |
| 3. Verificación de la edición | Conciliación de los 5.676 candidatos, recuentos frente al censo de ventanas, máscaras, disponibilidad no posterior a la decisión, ninguna fila de 2024 | Verificador por prefijos existente. Falta la pasada completa | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 4. Protocolo temporal | Ventanas anuales expansivas desde el primer año con etiquetas, validación interna al final de cada tramo de entrenamiento, evaluación del año siguiente, purga por intervalo real de etiquetas | Diseño y comprobaciones técnicas en el [protocolo v2](walk-forward-2000.md). Variantes de presupuesto A y B declaradas con su [recuento de trabajos](#variantes-de-presupuesto). Falta medir el caudal, elegir la variante y preparar las vistas reales | [#363](https://github.com/GonxKZ/mars-titan/issues/363) |
| 5. Entrenamiento base | Referencias, GRU episódica, Transformer compacto, núcleo Titans-MAC, MARS-TITAN con ampliaciones y CM-v1 | [Orquestación](#orquestación-de-los-brazos-con-entrenador) de las referencias neuronales y tabulares preparada y comprobada con dobles, sin ejecutar. Los entrenadores cronológicos de [Titans-MAC](../engineering/titans-chronological-trainer.md) y de la [GRU episódica](../engineering/candidate-chronological-trainer.md) están comprobados sin pasos de optimizador. La GRU episódica ya tiene entrada por ventana y traslado para B, pero las campañas declaradas aún no la incluyen | [#234](https://github.com/GonxKZ/mars-titan/issues/234), [#23](https://github.com/GonxKZ/mars-titan/issues/23), [#293](https://github.com/GonxKZ/mars-titan/issues/293) |
| 6. Postentrenamiento | Padre congelado, continuación supervisada, corrección residual y adaptadores, solos y combinados | Implementado sobre la edición estricta. Falta la edición con máscaras y la matriz de combinaciones | [#364](https://github.com/GonxKZ/mars-titan/issues/364), [#128](https://github.com/GonxKZ/mars-titan/issues/128) |
| 7. Refuerzo | Variantes de PPO, KLPO prioritario y Double DQN sobre entornos auditados | Controladores implementados sin ejecutar pasos. Auditoría de entornos en curso | [#137](https://github.com/GonxKZ/mars-titan/issues/137), [#365](https://github.com/GonxKZ/mars-titan/issues/365) |
| 8. Evaluación | MAE residual por sesión y métricas secundarias con incertidumbre por bloques | Métricas, calibración común CQR y [evaluación walk-forward](metrics.md#evaluación-walk-forward-de-la-edición-desde-2000) implementadas con pruebas técnicas y [configuración declarada](../../configs/evaluation/historical-masked-2000-comparison.json). Sin aplicar a esta edición. Las referencias neuronales y tabulares ya escriben las predicciones por fila de calibración y evaluación, y la orquestación publica el manifiesto de fuentes de cada ámbito. Faltan los productores de las demás familias | [#32](https://github.com/GonxKZ/mars-titan/issues/32), [#36](https://github.com/GonxKZ/mars-titan/issues/36) |

## Población y equidad

Todas las arquitecturas leen las mismas filas de la edición. Una fila existe aunque falten noticias, fundamentales o macro, porque la ausencia se marca y se rellena con cero de forma explícita. Ninguna variante puede descartar filas difíciles para mejorar su promedio.

Los bits de presencia forman parte de la entrada de todas las familias. Titans y la GRU candidata ya los recibían. Con la política con máscaras las referencias neuronales los fusionan con las modalidades y Ridge, HistGradientBoosting y XGBoost los reciben como cinco columnas más. Sin ese cambio, una diferencia de error podría deberse a que un modelo distingue un cero real de una ausencia y otro no.

La comparación estricta, que exige las cuatro modalidades completas y los 140 indicadores, se conserva como control separado. No se mezclan sus filas con las de la edición desde 2000.

## Validación temporal y sobreajuste

Cada ventana entrena con todo el pasado disponible hasta su corte, valida en el tramo final de ese pasado y evalúa el año siguiente. La purga elimina las filas cuya etiqueta madura después del corte. China no tiene etiquetas residuales antes de 2006 porque sus precios y el factor CSI300 empiezan ese año. Por eso las ventanas conjuntas empiezan en 2011, el primer año con tres años de etiquetas maduras en los dos mercados, y ninguna ventana admite un mercado vacío. Las ventanas solo de US evalúan desde 2005 y siguen entrenando con todo el pasado disponible. El [protocolo v2](walk-forward-2000.md) justifica la decisión.

La selección guarda el mejor estado según la validación temporal, con presupuesto fijo de 30 épocas, paciencia y mejora mínima declaradas en el protocolo. Los controles emparejados conservan el mismo número de actualizaciones. Si una parada independiente rompiera esa igualdad, se usa selección del mejor estado con presupuesto fijo. Los checkpoints de recuperación rotan con un límite pequeño y el mejor estado se guarda aparte, según [la política de checkpoints](../engineering/checkpoint-recovery.md).

Los brazos con memoria predicen cada tramo desde el estado inicial, tras observar las entradas de los 12 meses anteriores sin etiquetas. El calentamiento es el mismo para los cuatro controles de Titans-MAC y está declarado en sus recetas antes de entrenar. Ninguna etiqueta madura se escribe en la memoria rápida. La [política completa](../engineering/titans-chronological-trainer.md#política-de-memoria-en-inferencia) explica por qué no se encadena la memoria entre tramos.

El test de 2024 no participa en ninguna selección. Se abrirá una sola vez, con la configuración fijada, al final de la campaña.

## Familias entrenadas

| Familia | Identidad | Qué aísla |
| --- | --- | --- |
| Referencias | RNN, LSTM, GRU, DLinear, Transformer compacto, Ridge y XGBoost | Nivel de error sin memoria persistente |
| GRU episódica | Candidato con banco 128×256, referencia independiente | Memoria episódica sobre un codificador recurrente |
| Núcleo Titans-MAC | `transformer_direct`, `mac_disabled`, `mac_frozen`, `mac_online` | Cambio de codificador frente a memoria neuronal |
| MARS-TITAN con ampliaciones | Banco episódico, escritura M0 a M3, refinamientos K=1, 2 y 4, y las modificaciones del [documento de integración](system-integration.md), cada una desactivable | Aportación de cada ampliación, una cada vez |
| CM-v1 | B, B+C, B+M y B+C+M sobre la B elegida por protocolo | Control del radio numérico y consolidación, según [su especificación](../experiments/mars_titan_cm_v1/specification.md) |

No se ejecuta el producto cartesiano de todas las ampliaciones. Primero se fija la base y después se estudia un mecanismo cada vez, como establece el [documento de integración](system-integration.md).

## Orquestación de los brazos con entrenador

La campaña se declara en dos configuraciones que solo difieren en el presupuesto: la [variante A](../../configs/baselines/historical-masked-campaign-a.json) y la [variante B](../../configs/baselines/historical-masked-campaign-b.json). Brazos, semillas, ámbitos, protocolos y ventanas se leen de la [comparación walk-forward declarada](../../configs/evaluation/historical-masked-2000-comparison.json), así que los productores y la evaluación comparten una única definición. El plan está en `training/campaign_plan.py`, la ejecución en `training/masked_campaign.py` y las predicciones trasladadas en `training/carried_predictions.py`. Nada de esto se ha ejecutado con datos reales.

### Variantes de presupuesto

La variante A reentrena desde cero cada ventana anual, con su ajuste, su validación y su selección. La variante B reentrena desde cero la primera ventana de cada ámbito y después cada 36 meses. Las ventanas intermedias no se ajustan y se predicen con el estado seleccionado en la última ventana reentrenada, que llamamos ancla. Con ese paso las anclas de US evalúan 2005, 2008, 2011, 2014, 2017, 2020 y 2023, y las de CN y US+CN evalúan 2011, 2014, 2017, 2020 y 2023, un subconjunto exacto de las de US. A diferencia de la alternativa B del [protocolo](walk-forward-2000.md#coste-y-alternativas-de-presupuesto), que solo evaluaba las ventanas reentrenadas, esta variante evalúa todos los años.

| Elemento de B | Regla |
| --- | --- |
| Estado usado | El estado seleccionado en el ancla para cada brazo y semilla: el ganador de la búsqueda con la semilla 42 y el finalista de cada otra semilla. No se cambian pesos, normalizadores ni selección |
| Información del ancla | Termina al final de su validación, el 1 de octubre del año anterior a su evaluación, con etiquetas maduras antes de esa fecha. El plan exige que esa fecha no sea posterior al comienzo de la calibración de cada ventana trasladada, y la predicción lo comprueba otra vez con los manifiestos |
| Filas y purga | Las de la vista de la ventana trasladada, con la purga por intervalo de etiqueta en sus propias fronteras. Son exactamente las filas que esa ventana tiene en A |
| Calibración | La comparación ajusta la calibración común con las predicciones del modelo trasladado en el tramo de calibración de esa ventana (los tres meses anteriores al año evaluado) y la congela antes de leer la evaluación |
| Ajuste y validación de la ventana trasladada | No se usan |

El precio de B es la antigüedad del modelo. La información del ancla termina 3 meses antes de su propia evaluación, 15 meses antes en el primer año trasladado y 27 en el segundo. B responde por tanto a otra pregunta, el error con reentrenamiento trienal, y no debe mezclarse con A. Como todos los brazos siguen el mismo calendario, la comparación emparejada dentro de cada variante conserva las mismas filas y la misma antigüedad.

En cada ventana reentrenada, cada brazo neuronal ajusta los dos candidatos del diseño (índices 0 y 10) con la semilla 42 y un finalista por cada semilla restante. Ridge ajusta sus tres alfas con la semilla 42, la única que le asigna la comparación, y XGBoost ajusta sus doce configuraciones con la semilla 42 y un finalista con cada una de las otras dos semillas. En cada ventana trasladada se hace una predicción por brazo y semilla.

| Variante | Ámbito | Ventanas reentrenadas | Ventanas trasladadas | Ajustes | Predicciones trasladadas |
| --- | --- | ---: | ---: | ---: | ---: |
| A | US | 19 | 0 | 703 | 0 |
| A | CN | 13 | 0 | 481 | 0 |
| A | US+CN | 13 | 0 | 481 | 0 |
| A | Total | 45 | 0 | 1.665 | 0 |
| B | US | 7 | 12 | 259 | 228 |
| B | CN | 5 | 8 | 185 | 152 |
| B | US+CN | 5 | 8 | 185 | 152 |
| B | Total | 17 | 28 | 629 | 532 |

Ajustes por brazo y semilla en cada ámbito:

| Brazo y semilla | A, US | A, CN o US+CN | B, US | B, CN o US+CN |
| --- | ---: | ---: | ---: | ---: |
| Cada referencia neuronal, semilla 42 | 38 | 26 | 14 | 10 |
| Cada referencia neuronal, semillas 43 y 44 | 19 | 13 | 7 | 5 |
| Ridge, semilla 42 | 57 | 39 | 21 | 15 |
| XGBoost, semilla 42 | 228 | 156 | 84 | 60 |
| XGBoost, semillas 43 y 44 | 19 | 13 | 7 | 5 |

En B cada brazo y semilla añade 12 predicciones trasladadas en US y 8 en CN y en US+CN. Cada configuración declara `max_training_jobs` y `max_prediction_jobs` iguales a su plan, de modo que añadir brazos, semillas o candidatos exige cambiar la configuración y su huella. Superarlos detiene la comprobación con el recuento previsto y el límite. La búsqueda temporal de referencias ya no tiene un tope fijo de 512 ejecuciones: un plan de versión 4 puede declarar `max_runs`, y sin ese campo conserva el límite anterior.

La elección entre A y B no se toma aquí. Depende del caudal medido y se registrará en [#363](https://github.com/GonxKZ/mars-titan/issues/363) antes del primer entrenamiento, con la huella de la configuración elegida.

### Parada y pérdida

Los casos neuronales se construyen con el diseño de referencias y la regla `stopping_rule(protocol)`: presupuesto fijo de 30 épocas con el mejor estado, MAE residual por sesión en validación, paciencia 5 como diagnóstico y mejora mínima 0,00001. La campaña rechaza protocolos con reglas distintas entre ámbitos, y cada caso debe coincidir con la regla antes de planificarse.

La pérdida común de los brazos neuronales es la pinball de [`quantile_head_v1`](../engineering/quantile-head.md), decidida en [#22](https://github.com/GonxKZ/mars-titan/issues/22). En la campaña sustituye a la familia de pérdidas del diseño de referencias, que alterna MAE, MSE y Huber según el índice. Los índices 0 y 10 conservan su tasa de aprendizaje, anchura, profundidad y abandono, y solo cambian la cabeza y la pérdida. El término de la mediana es la pérdida absoluta, coherente con la métrica principal. La búsqueda de la familia de pérdidas sigue en los estudios escalares y en el [control de la cabeza](../../configs/baselines/quantile-head-control-us.json). Si ese control obligara a volver a la salida escalar, haría falta otra configuración de campaña con otra huella.

XGBoost no sigue la regla neuronal. Usa hasta 2.000 rondas con meseta de validación, un mínimo de 200 rondas, paciencia de 100 y la misma mejora mínima de 0,00001, sobre el mismo tramo de validación y con el MAE por sesión. Hay tres diferencias: la unidad es una ronda de boosting y no una época, se detiene en la meseta en lugar de agotar un presupuesto fijo, y el mejor modelo se sustituye con cualquier mejora estricta mientras la mejora mínima solo cuenta para la paciencia. No se alinea. El presupuesto fijo existe para que los controles neuronales emparejados apliquen el mismo número de actualizaciones, y XGBoost no forma parte de esos pares. Agotar siempre las 2.000 rondas, cada una con una lectura completa de páginas [estimada en 26 a 30 GB](../engineering/masked-tabular-comparators.md), multiplicaría el coste sin una comparación emparejada que lo justifique. Cambiar `BoostingSelection` cambiaría además la identidad de la ruta estricta. Ridge no tiene épocas y elige su alfa con el MAE por sesión de validación.

### Referencias tabulares

Ridge, HistGradientBoosting y XGBoost aceptan la retención reservada `heldout_full_train_sessions_v1`. Escriben validación, calibración y evaluación por fila con el mismo esquema que las referencias neuronales y resumen el ajuste sin tabla. Sin declararla conservan el recibo anterior de ajuste y validación.

El recorrido por ventanas no pasa por `posttraining/`. Tras [#386](https://github.com/GonxKZ/mars-titan/pull/386) la cola tabular de `posttraining/completion.py` lee la política, pero sigue esperando a una búsqueda neuronal completa sobre una sola vista, usa `run_tabular_search` y solo conserva ajuste y validación. Adaptarla habría exigido cambiar su contrato, así que la campaña llama directamente a los ejecutores tabulares.

HistGradientBoosting queda fuera del plan. Su ajuste concatena la matriz completa en memoria en `float64`, y con unos 15,4 millones de filas y 1.719 columnas serían unos 212 GB, frente a los 32 GB del equipo. Incluirlo exigiría muestrear filas, que rompe la población común, o un ajuste con memoria externa que aún no existe.

### Ejecución y recuperación

```bash
uv run --no-sync python scripts/run_masked_campaign.py check \
  --campaign configs/baselines/historical-masked-campaign-b.json
uv run --no-sync python scripts/run_masked_campaign.py prepare \
  --campaign <configuración> --parent <supervisión histórica> --output <vistas>
uv run --no-sync python scripts/run_masked_campaign.py run --campaign <configuración> \
  --views US=<vistas>/US --views CN=<vistas>/CN --views US+CN=<vistas>/US+CN --output <campaña>
uv run --no-sync python scripts/run_masked_campaign.py sources --campaign <configuración> \
  --views US=<vistas>/US --output <campaña> --scope US --comparison <comparación>
```

`check` valida y cuenta sin leer datos. `prepare` crea las vistas de cada ámbito desde la supervisión histórica con los protocolos v2 y la recuperación de fronteras anuales (`recover_annual_boundaries`). Para un solo mercado escribe antes la proyección de la supervisión en ese mercado. Las vistas se comprueban frente a la comparación: misma política, mismas ventanas, mismos protocolos, una sola edición y todos los tramos con filas. Una vista de otro ámbito, de otro protocolo o de otra política se rechaza antes de crear la salida.

`run` llama a `require_learning_allowed` antes de abrir fuentes y otra vez antes de cada trabajo. Si la protección vuelve a estar vigente a mitad de campaña, `LearningHoldError` la detiene antes del siguiente trabajo y el resumen queda en `blocked`. La salida guarda la identidad de la campaña (configuración, comparación, configuración tabular, huellas de vistas y código) y un recibo por trabajo con su identidad, el intento, el informe del ejecutor, el MAE de validación y, para calibración y evaluación, la huella del Parquet, el número de filas y una huella de filas y objetivos independiente del orden. Cada predicción debe caer en su tramo, no contener filas de 2024 y tener tantas filas como la vista. Todos los trabajos de una ventana deben dar la misma huella de filas y objetivos, y el primero que difiera detiene la campaña.

Al reanudar, un trabajo con recibo e identidad iguales no se repite, después de comprobar las huellas de sus artefactos. Los trabajos sin recibo se rehacen: las referencias neuronales y XGBoost continúan su último intento desde su punto de control, y Ridge y las predicciones trasladadas empiezan un intento nuevo. Una salida de otra campaña, otras vistas u otro código se rechaza. Los trabajos CUDA se ejecutan de uno en uno bajo una única reserva de la GPU. Los trabajos CPU usan la concurrencia `cpu_workers` de la configuración tabular. Hoy Ridge y XGBoost se ejecutan en CUDA, así que esa concurrencia solo se aplicará cuando se conecte un ejecutor tabular de CPU.

Cuando una semilla de un brazo ya tiene su predictor elegido en una ventana (el ganador de la búsqueda cuando terminan todos los candidatos, el finalista o la predicción trasladada), `run` escribe el recibo de ventana que leen los entornos de refuerzo, uno por mercado, en `windows/<ámbito>/<ventana>/<brazo>/seed-<semilla>/<mercado>.json`. Se construye con el contrato de `environments/walk_forward_receipt.py` y se valida con su `read_window_receipt` antes de escribirlo:

| Campo | Valor en la campaña |
| --- | --- |
| `protocol`, `fold` | Protocolo v2 del mercado y ventana de la comparación declarada |
| `parent` | Trabajo que ajustó el estado elegido y huella de ese estado. En una ventana trasladada es el trabajo del ancla, y la campaña comprueba que la predicción trasladada partió de ese mismo estado |
| `labels_used_until` | Microsegundo anterior al inicio de la evaluación |
| `predictions` | Filas y `prediction_fingerprint` de la mediana emitida en calibración y evaluación, solo con las filas de ese mercado |

`labels_used_until` es una cota y no la maduración exacta de la última etiqueta. La calibración común usa el tramo anterior a la evaluación y la purga por intervalo de etiqueta obliga a que todas sus etiquetas maduren antes del final del tramo, así que ninguna etiqueta usada en ajuste, selección o calibración madura después. En una ventana trasladada el modelo dejó de aprender antes, pero su calibración también usa ese tramo. Al reanudar, un recibo de ventana ya escrito debe coincidir con el que se deriva de los trabajos confirmados. No hay recibos reales porque la campaña no se ha ejecutado.

`sources` publica el manifiesto de un ámbito para `evaluation.walk_forward_comparison`. Elige para cada brazo, semilla y ventana el ganador de la búsqueda, el finalista o la predicción trasladada, vuelve a exigir las mismas filas en todos ellos y valida el manifiesto con `load_sources` antes de publicarlo. Con la comparación declarada de 23 brazos falla y nombra los brazos sin productor. Para evaluar antes solo las referencias haría falta declarar, antes de ver resultados, una comparación con esos brazos.

### Puntos de extensión y etapas posteriores

| Familia | Brazos de la comparación | Tarea | Falta |
| --- | --- | --- | --- |
| GRU candidata | `gru_episodic` | [#383](https://github.com/GonxKZ/mars-titan/issues/383) | Declarar la sección `episodic_gru` en A y B tras medir memoria y caudal en `cuda:0`. La [entrada por ventana](../engineering/candidate-chronological-trainer.md#ventanas-walk-forward-de-la-campaña) ya existe |
| Titans-MAC | `titans_transformer_direct`, `titans_mac_disabled`, `titans_mac_frozen`, `titans_mac_online` | [#23](https://github.com/GonxKZ/mars-titan/issues/23) | Conectar el entrenador cronológico a las vistas v2 y a `stopping_rule` |
| MARS-TITAN | `mars_titan_m0` a `mars_titan_m3`, `mars_titan_m1_k2`, `mars_titan_m1_k4` | [#366](https://github.com/GonxKZ/mars-titan/issues/366) | Ampliaciones sobre el núcleo con sus puntos de inserción |
| CM-v1 | `cm_v1_b`, `cm_v1_bc`, `cm_v1_bm`, `cm_v1_bcm` | [#293](https://github.com/GonxKZ/mars-titan/issues/293) | Brazos sobre la B fijada por protocolo |

Cada familia se conectará con su planificador y su ejecutor en el mismo registro. Hasta entonces el plan las informa como pendientes y no crea trabajos falsos. La GRU candidata ya está conectada: `masked_campaign.EXECUTORS` registra su ajuste por ventana y su traslado, y una sección opcional `episodic_gru` de la campaña declara su receta, la variante de cada brazo y la semilla de búsqueda. Sus predicciones contienen exactamente las filas de la vista y las de los demás brazos, comprobado con vistas del corpus técnico. Las campañas A y B declaradas no incluyen todavía esa sección. Declararla en B añadiría 51 ajustes y 84 traslados (680 y 616 en total) y en A 135 ajustes (1.800), y antes hay que elegir con una medida en `cuda:0` entre `accumulation_rows` y `recompute`, porque la extrapolación desde CPU del tramo completo supera los 8 GB con el universo completo. Mientras tanto, la comprobación sigue informando del brazo como pendiente en [#383](https://github.com/GonxKZ/mars-titan/issues/383).

El postentrenamiento con la [matriz de adaptadores](../../configs/posttraining/adapter-matrix-v1.json) de [#364](https://github.com/GonxKZ/mars-titan/issues/364) es una etapa posterior que partirá de los padres seleccionados en cada ventana. Solo está declarado. Faltan la ejecución desde la cola, la conexión con las ventanas walk-forward y el objetivo pinball para padres con cuantiles.

### Medición de caudal

```bash
uv run --no-sync python scripts/run_masked_campaign.py throughput \
  --campaign configs/baselines/historical-masked-campaign-a.json \
  --campaign configs/baselines/historical-masked-campaign-b.json \
  --views US=<vistas>/US --views CN=<vistas>/CN --views US+CN=<vistas>/US+CN \
  --first-view <vistas>/US/fold-000/manifest.json --batches 50 --warmup 5
```

La orden reserva la GPU con la regla de una sola carga y, para cada brazo neuronal y candidato, recorre lotes reales de la primera ventana con forward, pinball y backward. Libera los gradientes sin crear ningún optimizador y un gancho global rechaza cualquier paso durante la medición. Mide también la inferencia sobre la validación. Solo registra filas por segundo y memoria, nunca pérdidas ni errores. Con esos caudales estima las horas neuronales de cada variante y ámbito: cada ajuste suma 30 épocas de ajuste y validación más las predicciones finales, cada traslado suma calibración y evaluación, y los finalistas y traslados usan el candidato más lento. Supone el mismo caudal en todas las ventanas y no incluye esperas de disco ni reanudaciones. Ridge y XGBoost quedan como no medidos, porque medirlos ya sería ajustarlos. La orden no se ha ejecutado.

### Comprobaciones técnicas

Las pruebas de `tests/training/test_campaign_plan.py`, `test_masked_campaign.py`, `test_carried_predictions.py`, `test_campaign_throughput.py` y `test_tabular_retention.py` no ajustan modelos ni ejecutan pasos de optimizador y no usan la GPU. Comprueban los recuentos exactos de A y B por ámbito, brazo y semilla, la causalidad de cada traslado, la regla de parada y la pinball de cada caso, el rechazo de límites superados, de reglas mezcladas y de configuraciones alteradas, los trabajos lanzados con su vista, ventana, semilla y caso, la reanudación tras una interrupción simulada, el bloqueo al empezar y a mitad de campaña, el rechazo de vistas o políticas mezcladas, filas distintas, objetivos distintos y filas de 2024, la concurrencia CPU y la cola acotadas, los recibos de ventana validados con `read_window_receipt` (padre elegido, cota de la última etiqueta y huellas por mercado), su comprobación al reanudar, el rechazo de un traslado que no parte del estado del ancla, el manifiesto aceptado por `walk_forward_comparison` en US y US+CN y la medición sin cambiar pesos. Los ejecutores se sustituyen por dobles que escriben predicciones nulas con las filas exactas de cada vista, y las pruebas admiten la campaña con la protección temporal permitida de `learning_doubles`, como el resto de lanzadores con dobles. Las predicciones trasladadas se prueban en CPU con un ancla neuronal construida a mano con pesos iniciales y con una función tabular fija.

## Postentrenamiento

Cada postentrenamiento parte de un padre seleccionado en la misma ventana y se compara con ese padre congelado. Los adaptadores se colocan solos y en combinaciones de uno, dos o tres puntos de inserción, con el mismo presupuesto de actualizaciones y la misma validación. La corrección residual inicializada a cero se compara con la salida continua del padre, no solo con la mediana de una rejilla. Los objetivos ya derivados se describen en [adaptación predictiva](predictive-adaptation.md).

## Refuerzo

Los entornos consumen únicamente predicciones fuera de muestra del walk-forward, identificadas por los [recibos de ventana](#ejecución-y-recuperación) de la campaña. La ejecución usa el precio posterior a la decisión, con costes y deslizamiento declarados y límites de posición. Un agente escrito a mano que intente leer información futura debe fallar o no obtener ventaja. Los resultados sobre entornos sintéticos no se presentan como resultados sobre FinMultiTime. El controlador KLPO terminal se describe en [su documento de ingeniería](../engineering/terminal-klpo-updates.md).

## Métricas

La métrica principal es el MAE residual por sesión. Primero se promedian los activos de un mismo mercado e instante y después se aplica la ponderación temporal y entre mercados. Se registran también MSE y RMSE, acierto de dirección con convención de empates, correlación de rangos por sesión y, cuando la salida lo permita, pérdida pinball, cobertura y anchura de cuantiles con y sin calibración común. La diferencia frente a una referencia se informa como Delta_error = MAE_variante − MAE_base y como porcentaje 100·(MAE_base − MAE_variante)/MAE_base, con intervalos del 95 % por bloques temporales. El detalle está en [métricas](metrics.md).

## Cómputo

La campaña se ejecuta en una RTX 4070 Laptop de 8 GB con el perfil de energía de ahorro, que el equipo necesita para no apagarse por temperatura. En esas condiciones la GPU trabaja a unos 1.305 MHz con limitación térmica. La auditoría de preparación estima unas 85 h por familia, configuración y semilla si se reentrena cada ventana anual completa con 30 épocas. Es una hipótesis basada en caudales de ediciones anteriores. El presupuesto definitivo se fijará con el caudal medido en la primera ventana mediante la [orden de medición](#medición-de-caudal), antes de lanzar la campaña, y será el mismo para los brazos emparejados.

## Decisiones pendientes antes de entrenar

- Variante A o B de la [orquestación](#variantes-de-presupuesto), con el caudal medido en la GPU y la huella de la configuración elegida ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).
- Comparación parcial con las referencias, si se quiere evaluarlas antes de conectar las demás familias.
- Revisar la configuración de evaluación declarada antes de ver resultados: familias de contrastes, base de los refinamientos K, mínimo de activos del Rank IC y longitud de bloque ([#32](https://github.com/GonxKZ/mars-titan/issues/32)). La [cabeza común](../engineering/quantile-head.md) y su calibración CQR ([#22](https://github.com/GonxKZ/mars-titan/issues/22)) están implementadas y el control de la cabeza sobre el Transformer compacto está declarado sin ejecutar.
- Semillas fijas y margen mínimo relevante de error, registrados antes de ver resultados.
- Política de retención de predicciones y checkpoints según el disco disponible.
- Memoria de Titans-MAC para la campaña. La [memoria con residual y LayerNorm](../engineering/titans-mac-output-scale.md) de la sección 3.3 del artículo mantiene la escala de la salida de MAC en fixtures, pero cambia la identidad de los cuatro brazos y queda propuesta en [#27](https://github.com/GonxKZ/mars-titan/issues/27).
- `accumulation_rows` según los activos por instante. Sin acumulación, el grafo de un tramo de `mac_online` con más de unos mil activos supera la memoria de la GPU según la [estimación medida en CPU](../engineering/titans-chronological-trainer.md#memoria-del-tramo-y-acumulación-por-bloques).
