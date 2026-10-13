# Entrenador cronológico de la GRU candidata

`training/candidate_run.py` ajusta la [GRU candidata con banco episódico](gru-financial-sessions.md) recorriendo en orden los instantes de decisión de la edición desde 2000. Es un brazo independiente de la comparación, con identidad propia, y no sustituye a Titans-MAC ni a MARS-TITAN. Está implementado y comprobado técnicamente en CPU sin pasos de optimizador. No se ha ejecutado ningún entrenamiento, piloto ni evaluación científica: el bloqueo de aprendizaje sigue vigente y el test de 2024 permanece cerrado.

## Cómo se ajusta el módulo nativo

El cálculo de la candidata vive en `Candidate` (C++ con LibTorch). Se compararon tres formas de ajustarlo:

| Opción | Qué exige | Decisión |
| --- | --- | --- |
| (a) Optimizador de PyTorch sobre el módulo nativo a través del enlace | Que el enlace devuelva tensores de la misma instalación de PyTorch conectados al grafo | Elegida |
| (b) Réplica PyTorch del mismo cálculo con paridad frente al nativo | Duplicar GRU, lectura top-k estable, normalización escalada, refinador y cabeza, y mantener la paridad en cada cambio | Descartada |
| (c) Bucle nativo con `torch::optim` | Llevar a C++ lector de corpus, banco, cola, checkpoints y selección, que hoy están en Python | Descartada |

La opción (a) funciona sin cambiar el código nativo. `_episodic_native` se compila contra la misma instalación de PyTorch, `named_parameters()` devuelve en cada llamada los mismos objetos Python de los 25 tensores registrados y `forward` conserva el grafo cuando el modo con gradiente está activo. Una sonda con las dimensiones reales (precios 64×5, noticias 384, gráficos 512, fundamentales 45 y macro 420), 128 filas, FP32 y 16 episodios obtuvo gradientes finitos y no nulos en los 25 tensores, sin aplicar ningún paso. Las salvedades del C++ también se respetan: el top-k queda fuera de autograd y con un único vecino el gradiente de la consulta es cero.

La opción (b) añadiría una segunda implementación del brazo cuya identidad habría que demostrar equivalente en cada cambio. Las pruebas nativas ya contrastan la GRU con sus ecuaciones explícitas y la consulta con diferencias finitas, así que la réplica no aporta una comprobación que falte. La opción (c) movería a C++ todo el recorrido sin un beneficio medido. Además, sus pasos no pasarían por el gancho global de `torch.optim` y necesitarían el lector nativo de la protección. El coste por fila está en la GRU de ATen y en su backward, que ya son nativos en (a).

Los parámetros solo cambian mediante el optimizador. Las proyecciones fijas del codec (rasgos 256 y claves 128) no son parámetros, así que el contenido del banco no depende de lo aprendido. Solo cambia cómo se consulta y cómo se proyectan sus valores.

## Recorrido y orden de cada evento

El entrenador reutiliza el índice de [observaciones financieras](financial-session-v2.md) y su lector por bloques. Dentro de un evento el orden es fijo:

1. Se toma la instantánea del banco confirmado antes de las etiquetas del evento.
2. Se resuelven las etiquetas que maduran en ese instante contra la mediana realmente emitida. Su error entra en `SessionErrors`, su grafo entra en la pérdida del tramo y, con M1, se prepara su admisión.
3. Si el tramo ya contiene `update_instants` instantes de decisión, o el evento cierra la fase, se calcula la pérdida, se recorta el gradiente y se llama al optimizador. Después se descartan todos los grafos.
4. Se predicen las entradas del instante con los parámetros vigentes y la instantánea del paso 1.
5. Se publican en el banco las admisiones preparadas en el paso 2.

Es el mismo orden que el entrenador de Titans-MAC para pérdida y actualización, y el mismo que `FinancialSession` para el banco: las predicciones de un instante no ven las etiquetas que maduran en ese instante. Las admisiones siguen el orden canónico del ejecutor nativo (maduración, decisión y flujo) y reciben IDs consecutivos desde `seen + 1`. Las claves y valores admitidos son los bytes del codec fijo CPU de la fila emitida y la etiqueta se guarda en FP64.

La GRU reinicia su estado en cada ventana, así que no hay BPTT entre instantes. `update_instants` solo fija la cadencia de actualización y coincide con `truncation` de Titans-MAC para que ambos brazos actualicen en los mismos eventos. Una etiqueta cuya predicción pertenece a un tramo ya actualizado no tiene grafo y se cuenta en `labels_without_graph`. Con etiquetas de la sesión siguiente vale cero en las pruebas.

El calentamiento no predice ni cambia el estado de la candidata, porque la GRU no conserva estado entre instantes y el banco solo admite etiquetas maduras de predicciones emitidas. Aun así, la ventana se recorre con el mismo calentamiento de entradas que Titans-MAC, MARS-TITAN y CM-v1: `walk_forward.warmup_months` de la receta vale 12 y las fases salen de `walk_forward_phases.window_phases`, la misma función que usan esas familias. Así todas las familias con memoria observan la misma ventana de información, sin etiquetas anteriores al tramo medido y sin pasar del origen del ajuste. El banco empieza vacío en cada pasada de ajuste, validación, calibración y evaluación, como pide el [protocolo walk-forward v2](../research/walk-forward-2000.md#estado-y-calentamiento-por-ventana). Un calentamiento episódico con etiquetas anteriores al tramo sería otra regla y necesitaría su propio contraste.

## Banco, refinamientos y política

| Política | Lectura | Escritura | Parámetros sin gradiente |
| --- | --- | --- | --- |
| M0 | Memoria vacía, el refinador se conserva | Ninguna | `query_*` y `value_*`, fuera del optimizador |
| M1 | Instantánea del banco confirmado | Reservorio causal v2 con etiquetas maduras | Ninguno una vez admitidos dos episodios |

K = 1 es la configuración principal. K = 2 y K = 4 repiten lectura y refinamiento sobre la misma instantánea y no escriben en el banco ni cuentan como actualizaciones de memoria. M2 y M3 no están definidos sobre la geometría 128×256 de este banco y siguen fuera.

## Pérdida y cabeza

La cabeza es la de `Candidate::quantiles`, que la PR #381 reprodujo como `quantile_head_v1` con paridad exacta de salidas y gradientes en FP32 y FP64. La parametrización coincide (mediana libre e incrementos softplus acumulados hacia cada cola), así que el entrenador no cambia la cabeza común. La pérdida es `pinball_loss` de `models/quantile_head.py` sobre los cinco cuantiles y la selección usa el MAE por sesión de la mediana emitida.

La variante `m1_k1_l1_median` declara la salida escalar de retroceso de [#22](https://github.com/GonxKZ/mars-titan/issues/22): pérdida L1 sobre la mediana. Las pruebas comprueban que con ella solo la fila de la mediana de `head_weight` recibe gradiente.

## Roles de parámetros

| Rol | Parámetros |
| --- | --- |
| `encoder` | GRU de precios, proyecciones de noticias, gráficos, fundamentales y macro, fusión y estado inicial |
| `episodic_read` | Consulta y proyección de valores leídos (`query_*`, `value_*`) |
| `refiner` | `update_*` y `step_logit` |
| `head` | `head_weight` y `head_bias` |

El optimizador debe cubrir exactamente los roles de la política. Con M0, la lectura episódica queda fuera porque nunca recibe gradiente.

## Selección y presupuesto

`load_recipe` lee [`configs/candidate/chronological-training.json`](../../configs/candidate/chronological-training.json) y el protocolo que referencia, y rechaza la receta si épocas, paciencia, mejora mínima o modo de parada difieren de `stopping_rule(protocol)`. La receta propuesta recorre 30 épocas con `fixed_budget`, paciencia 5 como diagnóstico y `min_delta` 1e-5, y conserva el mejor estado. Recorte, lote de 128 activos y cadencia de 8 instantes coinciden con la receta de Titans-MAC. Las semillas son las del protocolo. Las de rasgos y claves del codec quedan fijas (43 y 44) para que el banco tenga la misma geometría en todas las semillas.

La sección `walk_forward.search_cases` declara los mismos dos casos de búsqueda que Titans-MAC y el lector de MARS-TITAN, `lr1e-4` y `lr1e-3`, con la regla común de `training/search_cases.py`: de uno a tres casos distintos que solo cambian `learning_rate` o `max_grad_norm`, ausentes de la receta base, y ninguna variante puede cambiarlos. `load_recipe(path, variant=..., search_case=...)` exige el caso cuando la receta los declara. Antes la candidata ajustaba un único caso con tasa 1e-3, así que la búsqueda era desigual frente a las demás familias neuronales.

La receta se identifica con el brazo `gru_episodic` de la [comparación walk-forward declarada](../../configs/evaluation/historical-masked-2000-comparison.json), que corresponde a la variante principal `m1_k1`. Las variantes `m0_k1`, `m1_k2`, `m1_k4` y `m1_k1_l1_median` son controles propios de este brazo y esa comparación todavía no las declara.

La receta de Titans-MAC declara hoy `min_delta` 0, distinto del protocolo v2. No se ha cambiado aquí, pero la comparación emparejada exige igualarlo antes de entrenar ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).

## Checkpoints y reanudación

`save_training_state` conserva los dos estados de recuperación más recientes y el mejor estado aparte. Un estado de recuperación incluye parámetros, huella de parámetros, optimizador, RNG, cursor, selección, historial y, dentro de un ajuste, la cola pendiente con sus claves y valores del codec, el banco con su RNG de reservorio, la admisión preparada del evento, los errores y los contadores.

La barrera está justo después de una actualización, sin grafos vivos. En ese punto ya se han resuelto las etiquetas del evento, pero sus predicciones y su admisión siguen pendientes. Por eso el estado guarda el banco previo y la admisión preparada por separado, y al reanudar se predice con el banco previo antes de publicar la admisión. La validación no se reanuda a mitad, porque es determinista y se repite desde su inicio. Los parámetros recuperados se copian en el módulo vigente, de modo que el optimizador conserva sus referencias, y se comprueba su huella.

`run` llama a `require_learning_allowed()` antes de abrir fuentes o crear salidas, con cualquier optimizador, y se detiene con `LearningHoldError` mientras la protección local no declare `training_allowed` verdadero. El entrenador no usa `torch::optim` en C++, así que no necesita el lector nativo de la protección. `evaluate` y `restore_selected` son inferencia congelada y no se bloquean.

## Predicciones

Tras la última época se carga el mejor estado y se escriben `validation-predictions.parquet` y, si se declaran sus fases, `calibration-predictions.parquet` y `evaluation-predictions.parquet`. El esquema es el de las referencias neuronales (`sample_id`, `asset_id`, `market`, `prediction_at`, `target`, `prediction` y `zero`) más `QUANTILE_COLUMNS`. `prediction` y `quantile_0500` tienen los mismos bits y `ForecastPanel.from_arrow` lee la tabla con sus cinco niveles. Solo se escriben las filas cuya etiqueta madura dentro de la fase.

`restore_selected` carga el mejor estado de una ejecución completa en un adaptador con el mismo contrato y comprueba la huella. Ese adaptador, congelado, sirve a `FrozenCandidateConsumer`. Con `carried=True` admite un adaptador de otra ventana de la misma edición: solo pueden cambiar la huella de la vista y la del índice de entrada, y deben coincidir representación, configuración, codec, binario y código.

La parte del recorrido que no ajusta (banco, instantánea, predicción, etiquetas, admisión y `evaluate`) está en `CandidateChronologicalPredictor`. `CandidateChronologicalTrainer` la extiende con el optimizador, la pérdida, las actualizaciones, los checkpoints y `run`. La división no cambia el comportamiento del entrenador y permite predecir una ventana posterior sin crear un optimizador.

## Paridad con la inferencia congelada

La GRU de ATen en CPU no produce los mismos bits cuando sus pesos requieren gradiente, aunque el modo sin gradiente esté activo. La diferencia medida fue de hasta 2,2e-16 en FP64 sobre el corpus técnico. Por eso la evaluación desactiva temporalmente `requires_grad` en los parámetros y lo restaura al terminar. Con ello la validación del entrenador coincide bit a bit con una repetición independiente que usa `FrozenCandidateConsumer`, su codec y un `CandidateEpisodeBank` propio, en M0 y M1 con K = 1 y 2. Las pruebas existentes de `FinancialSession` ya comprueban que la sesión emite lo mismo que ese consumidor con la instantánea previa a las etiquetas. No se ejecutó la sesión completa sobre el corpus técnico porque su verificador de prefijos necesita una edición materializada cuyo fixture estima objetivos residuales.

`FrozenCandidateConsumer`, `FinancialSession`, el banco y el código nativo no se han modificado.

## Ventanas walk-forward de la campaña

`training/candidate_walk_forward.py` conecta la candidata con la [campaña con máscaras](../research/training-campaign-2000.md#orquestación-de-los-brazos-con-entrenador) mediante dos entradas que siguen el contrato de trabajo de `masked_campaign.py`.

`fit_window` recibe la vista v2 de una ventana, la receta, la semilla, los meses de calentamiento y el identificador del trabajo. Comprueba la protección antes de abrir la vista, prepara en `indices/<tramo>` el índice de observaciones de los cuatro tramos y construye el adaptador de la semilla con las semillas fijas del codec. Las fases salen de `window_phases` con el calentamiento de la receta: el ajuste empieza en el origen y cada tramo medido observa antes las entradas de los 12 meses previos, sin pasar del origen. Se exige que los límites de decisión coincidan con los de la vista en todos sus mercados, y una vista con purga por sesiones se rechaza. El entrenador escribe en `run/` y predice validación, calibración y evaluación con el mejor estado. Después se comparan las filas de cada tramo con las de la vista y se escribe `window.json` con la vista, la receta, la ejecución, el estado elegido (`checkpoint`), las fases, las predicciones, los recibos y la política del banco. Si la salida ya existe, reanuda los índices confirmados y la ejecución desde su último checkpoint, y una ejecución completa se devuelve sin repetir nada.

`carry_window` predice la calibración y la evaluación de una ventana posterior con el estado elegido en su ancla, sin ajustar nada. `carried_window` de `carried_predictions.py` exige la misma edición y el mismo protocolo, y que la validación del ancla termine antes de la calibración trasladada. Se comprueban las huellas de `window.json`, del informe de la ejecución y del estado elegido, el adaptador se construye para la nueva ventana con la configuración del ancla y `restore_selected(..., carried=True)` carga sus parámetros. El recibo `carry.json` se escribe con `_receipt` del mismo módulo e incluye la huella del estado del ancla, la de sus parámetros y los meses entre el final de la información del ancla y la evaluación trasladada.

### Banco y cola al cruzar de ventana

| Estado | Al pasar del ancla a la ventana trasladada |
| --- | --- |
| Parámetros elegidos en el ancla | Se conservan sin cambios |
| Proyecciones fijas del codec y receta (admisión, K, capacidad, semilla y bloques del banco) | Se conservan |
| Banco episódico y admisión preparada | Se reinician. El banco empieza vacío en cada tramo |
| Predicciones a la espera de etiqueta | Se descartan al cerrar cada tramo |
| Errores por sesión y estado de la GRU | Se reinician |

No hay calentamiento con etiquetas del ancla ni de tramos anteriores. El traslado repite el calentamiento de entradas que registra el ancla en `bank_policy.warmup_months`. Cada tramo solo admite etiquetas que maduran dentro de él, después de emitir su predicción, y cada predicción lee la instantánea previa a las etiquetas de su instante. Es la regla que ya siguen la validación, la calibración y la evaluación de una ventana ajustada, así que una ventana trasladada solo se diferencia de una ajustada en los parámetros. Calentar el banco con etiquetas recientes podría ayudarle en los años intermedios, pero cambiaría también la regla de las ventanas ajustadas y necesitaría su propio contraste. La política se guarda como `bank_policy` en `window.json` y en `carry.json`.

### Filas y recibos

`check_view_rows` compara cada tramo predicho con las filas que declara la vista (mercado, activo, instante y objetivo de sus etiquetas aceptadas) y falla indicando cuántas faltan, cuántas son ajenas, cuántas se repiten y cuántas tienen otro objetivo. El lector de observaciones predice todas las muestras del tramo, pero solo escribe las filas cuya etiqueta madura dentro de él, que en una vista v2 son exactamente las de la partición.

Cada entrada escribe en `receipts/<mercado>.json` el recibo walk-forward de cada mercado con el contrato de `environments/walk_forward_receipt.py`, con calibración y evaluación como `masked_campaign.publish`. El padre es el trabajo ajustado con la huella de su estado elegido, y en un traslado es el ancla. `labels_used_until` es la maduración medida de las etiquetas que fijaron el ajuste, la selección y la calibración, con la regla `label_maturity.window_labels_until` que usa también `masked_campaign.publish`. En una ventana ajustada se leen los tramos de ajuste, validación y calibración de su vista. En un traslado se leen esos tramos en la vista del ancla y se añade la calibración de la propia ventana. `read_window_receipt` rechaza el recibo si ese instante alcanza la evaluación. Durante la evaluación el banco sigue admitiendo etiquetas que maduran antes de cada predicción.

### Registro en la campaña

`masked_campaign.EXECUTORS` registra `("episodic_gru", "fit")`, que reanuda su intento, y `("episodic_gru", "carry")`, que empieza uno nuevo. Ambos llaman a `run_job` en `cuda:0` y no usan el lote de la campaña, porque la candidata agrupa por `block_rows` según su receta. `confirm` decide si lee los cuantiles por la salida declarada del brazo en la comparación, no por su familia. `campaign_plan.py` acepta una sección opcional `episodic_gru` con la receta, la variante de cada brazo y la semilla de búsqueda:

```json
"episodic_gru": {
  "recipe": "../candidate/chronological-training.json",
  "arms": {"gru_episodic": "m1_k1"},
  "search_seed": 42
}
```

El planificador la valida sin importar PyTorch (nombre de la receta, política, cabeza, variante, semillas, regla de parada del protocolo y tantos casos de búsqueda como índices ajusta cada referencia neuronal). Cada ventana reentrenada tiene dos búsquedas con la semilla de búsqueda, una por caso, y un finalista por cada semilla restante con el caso elegido por validación. Cada ventana trasladada tiene una predicción por semilla. El caso guarda la ruta y la huella de la receta, la variante, la semilla y el caso de búsqueda, y `campaign_case` la vuelve a leer con `load_recipe` antes de ajustar. Devuelve la receta, las opciones del modelo y los meses de calentamiento. Si la campaña declara una [parada temprana](../research/walk-forward-2000.md#parada-temprana-opcional), el caso lleva además `stopping_rule` y la receta la aplica con la misma métrica.

Las configuraciones A y B aún no incluyen la sección. La de A v2 sí la declara. La campaña elegida, la A, la incorpora en su [declaración ampliada](../research/training-campaign-2000.md#declaración-preparada-de-las-familias-pendientes), donde añade 180 ajustes con sus dos casos de búsqueda. En B añadiría 68 ajustes y 84 traslados. El tramo completo no cabe en 8 GB con el universo completo, así que la receta de la campaña fija `accumulation_rows=128` con bloques de 128 flujos y sin `recompute` desde [#482](https://github.com/GonxKZ/mars-titan/pull/482), con la medida en `cuda:0` sobre US+CN fold-012. La [orden de medición](../research/training-campaign-2000.md#medición-de-caudal) de la campaña puede comparar además las cuatro combinaciones con su caudal. Mientras `memory_options` de A v2 no iguale ese valor, la comprobación de la campaña sigue informando del brazo como pendiente.

## Coste y memoria medidos

Una sonda en CPU con dos hilos, FP32, dimensiones reales, 128 filas, 16 episodios y K = 1 midió 0,055 s de forward con grafo, de 0,074 a 0,079 s de backward y 264 KB de tensores guardados por fila (almacenamientos únicos sin parámetros). Con la receta de #389 cada predicción conserva su grafo hasta el paso del tramo, que llega tras `update_instants` instantes. El pico crece con los activos de cada instante multiplicados por esos 8 instantes.

## Memoria del tramo: acumulación y recomputación

La receta declara dos opciones, desactivadas por defecto para conservar la paridad exacta con #389:

| Opción | Qué hace | Identidad y gradiente |
| --- | --- | --- |
| `accumulation_rows` | Al madurar las etiquetas de un evento calcula el backward de su pérdida sumada, por grupos de bloques de predicción completos de hasta ese número de filas (como mínimo `block_rows`). El paso del tramo divide el gradiente por el total de etiquetas, recorta y llama una vez al optimizador | Cambia `loss_reduction`. Solo cambian el orden de las sumas y la división final |
| `recompute` | Ejecuta cada bloque de ajuste con `torch.utils.checkpoint` sin reentrada. Guarda sus entradas y repite el forward durante el backward con los mismos parámetros, la misma instantánea y el mismo K | Cambia la receta. En las comprobaciones el gradiente coincide bit a bit con #389 |

En este entrenador el forward ocurre al predecir y la pérdida solo existe cuando madura la etiqueta, en un evento posterior. Por eso la acumulación agrupa al madurar los bloques formados al predecir, y el forward por bloques de `block_rows` filas no cambia. Cada grupo es una ponderación de sus filas sobre el total del tramo, que solo se conoce en el paso, así que la división se aplica una vez al gradiente acumulado. Un grupo nunca reparte un bloque, para no recorrer su grafo dos veces. Las predicciones, la instantánea de cada instante y la admisión al final del evento no cambian, y las pruebas lo comprueban sobre el registro de auditoría.

Un bloque solo libera su grafo cuando han madurado todas sus filas. Mientras tanto se conserva con `retain_graph`, de modo que una etiqueta que madure más tarde sigue entrando en la pérdida. La contrapartida es que una fila sin etiqueta (varianza de mercado nula, sesión siguiente ausente o historia insuficiente) retiene su bloque completo hasta el paso del tramo. En el corpus técnico una sola fila sin etiqueta mantuvo vivo un bloque de 128 filas. Liberar antes el grafo exigiría saber al predecir si la etiqueta llegará, y eso depende de datos posteriores. La recomputación evita el problema, porque un bloque retenido solo conserva sus entradas.

### Medidas en CPU

[`candidate_memory_check.py`](../../tests/training/candidate_memory_check.py) recorre una época completa sin pasos sobre el corpus técnico sintético con dimensiones reales (precios 64×5, noticias 384, gráficos 512, 45 fundamentales y 420 macro), FP32, `update_instants` 8, `block_rows` 128 y un banco de 1.024 episodios. El pico es el máximo neto del asignador de ATen en CPU durante la ejecución, leído de los eventos de memoria del perfilador de PyTorch. Incluye activaciones, entradas, recomputaciones, gradientes y temporales, pero no la memoria de Arrow ni de NumPy.

| Configuración | 128 activos | 256 activos | Pendiente por activo |
| --- | --- | --- | --- |
| Tramo completo (#389) | 293,5 MB | 564,5 MB | 2,12 MB |
| Acumulación con 128 o 1.024 filas | 88,8 MB | 122,8 MB | 0,27 MB |
| Recomputación | 74,1 MB | 82,4 MB | 65 KB |
| Acumulación y recomputación | 59,3 MB | 60,4 MB | 8 KB |

Los picos de la acumulación incluyen el bloque retenido por la fila sin etiqueta del corpus. El tamaño de grupo no cambia el pico, porque lo determinan los bloques vivos del instante. Frente al tramo completo, la mayor diferencia absoluta del gradiente fue 6,0e-8 con la acumulación, sobre un gradiente máximo de 0,1. Con la recomputación los gradientes coincidieron bit a bit en ambos tamaños.

El coste de la recomputación se midió sobre un bloque de 128 filas, con la mediana de cinco repeticiones: 0,055 s de forward y backward sin recomputar frente a 0,095 s con ella, y 0,021 s de forward sin grafo. Una medición anterior dio 0,068 s frente a 0,093 s. Ambas se hicieron con otros procesos en curso, así que el sobrecoste del ajuste queda entre un 36 % y un 74 %. Los tiempos de la época completa (de 16 a 40 s según el tamaño) incluyen validación, predicciones y lectura, y su dispersión no permite atribuir diferencias.

### Extrapolación a la GPU

Con N filas por evento, las medidas anteriores dan aproximadamente:

| Configuración | Memoria en CPU | N = 5.676 |
| --- | --- | --- |
| Tramo completo | 23 MB + 2,12 MB × N | 12,0 GB |
| Acumulación sin bloques retenidos | 21 MB + 264 KB × N | 1,5 GB |
| Recomputación | 66 MB + 65 KB × N | 0,43 GB |
| Acumulación y recomputación | 58 MB + 8 KB × N | 0,11 GB |

N = 5.676 es una cota superior, porque cada evento contiene un solo mercado. Es una estimación en CPU, contrastada después con la [medida en `cuda:0`](#medida-en-cuda0). La GRU de cuDNN guarda sus propios búferes, el contexto CUDA y los espacios de trabajo de cuBLAS y cuDNN no se cuentan como tensores y el asignador con caché fragmenta la memoria. Con esas salvedades, el tramo completo no cabe en los 8 GB de la RTX 4070 Max-Q, la acumulación cabe si pocos bloques quedan retenidos y la recomputación cabe con margen cualquiera que sea el patrón de etiquetas.

### Medida en `cuda:0`

El 9 de octubre se midió el pico neto del asignador de PyTorch durante un ajuste sin pasos, con la receta propuesta en FP32 (`update_instants=8`, `block_rows=128`, banco de 1.024) y dimensiones reales de entrada ([recibo](../../reports/engineering/cuda-checks-20261009/candidate-memory-cuda.json)):

| Activos | Sin acumulación | `accumulation_rows=128` | `accumulation_rows=1024` | Recomputación | Ambas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 357,9 MiB | 156,0 MiB | 156,0 MiB | 144,9 MiB | 148,2 MiB |
| 256 | 622,1 MiB | 189,5 MiB | 191,6 MiB | 151,5 MiB | 149,3 MiB |
| 512 | 1.157,8 MiB | 256,6 MiB | 258,7 MiB | 164,7 MiB | 150,9 MiB |

Sin acumulación el pico crece unos 2,1 MiB por activo entre 256 y 512, y con `accumulation_rows=128` unos 0,26 MiB. Una extrapolación lineal, no medida, sitúa el ajuste sin acumulación de 4.000 activos por encima de 8 GiB y con `accumulation_rows=128` en torno a 1,1 GiB. La recomputación de un bloque de 128 filas pasa de 5,6 a 8,3 ms en forward y backward y da gradientes idénticos bit a bit. La acumulación cambia el orden de suma y difiere hasta 9·10⁻⁸. La receta de la campaña usa `accumulation_rows=128` sin recomputación.

## Comprobaciones

Las pruebas de `tests/training/test_candidate_run.py` usan el corpus técnico cronológico con etiquetas declaradas, sin estimar objetivos, y un optimizador que solo registra llamadas y gradientes. Cubren:

- Gradientes finitos en cada parámetro de la política, con algún gradiente no nulo por parámetro, para M1 con K = 1, 2 y 4 y para M0, donde la lectura no recibe gradiente.
- Paridad bit a bit de la validación con la repetición mediante `FrozenCandidateConsumer`, incluidos los cuantiles escritos en Parquet.
- Etiquetas solo tras madurar, cada predicción leyendo exactamente las admisiones anteriores a su instante, orden e IDs de admisión del ejecutor y ningún grafo perdido con etiquetas de la sesión siguiente.
- Predicciones, admisiones y gradientes pasados iguales cuando se alteran entradas, precios y etiquetas posteriores a un corte.
- Reanudación con la misma secuencia de predicciones, admisiones, actualizaciones, gradientes, historial y Parquet que una ejecución continua, y recuperación de un estado construido a mano con banco ya sustituido, admisión preparada y cola.
- Copia de los parámetros guardados sobre valores escritos a mano, recorte de gradiente y contrato del Parquet con `ForecastPanel`.
- Gradiente de la cabeza con pinball en las cinco filas de `head_weight` y solo en la de la mediana con la variante L1, y descarte contado de una etiqueta cuyo grafo ya no existe.
- Escritura de las predicciones de validación, calibración y evaluación con el fixture temporal histórico, cada una dentro de su partición.
- Rechazo con la protección temporal bloqueada antes de crear salidas, con el optimizador de registro y con AdamW, y rechazo de políticas estrictas, vistas de otro corpus, particiones mal declaradas, parámetros congelados, ámbitos vacíos y cambios de receta al reanudar.

`tests/training/test_candidate_accumulation.py` comprueba las dos opciones de memoria:

- Gradientes de la acumulación con 2, 4 y 1.024 filas iguales a los del tramo completo en FP64 (rtol 1e-12, atol 1e-15, con una diferencia máxima observada de 4,2e-17) y en FP32 (rtol 1e-5, atol 1e-7, con 7,5e-9), y gradientes de la recomputación iguales bit a bit. Las predicciones, etiquetas, admisiones y pasos del registro coinciden en todos los casos.
- Un bloque cuyas filas maduran en eventos distintos conserva su grafo hasta la última, con y sin recomputación, y los grupos nunca reparten un bloque.
- Las filas de un bloque no ven las demás: alterar otras filas no cambia ni la salida ni el gradiente de la primera, con la misma instantánea para todas.
- La acumulación reduce el pico de tensores guardados y la recomputación deja fuera de los bloques solo la pila y la pérdida.
- Reanudación de una ejecución con acumulación, con y sin recomputación, igual a la continua.

`tests/training/test_candidate_walk_forward.py` comprueba las ventanas sobre vistas v2 del corpus técnico (un activo por mercado), con un optimizador que solo registra gradientes. El de las ventanas ancla desplaza además una vez los pesos al construirse, sin gradientes, para que su estado elegido se distinga de una inicialización nueva:

- El ajuste de una ventana escribe validación, calibración y evaluación con el esquema común y tantas filas como la vista, ninguna de 2024, el estado elegido con su huella y un recibo por mercado válido con `read_window_receipt`. El estado elegido es el inicial, porque ningún paso cambia pesos.
- Las filas predichas tienen la misma huella de filas y objetivos y los mismos `sample_id` que el lector por lotes que usan los demás brazos.
- La comprobación de filas cuenta por separado una fila ausente, una ajena, una repetida y una con otro objetivo, y también un cambio de mercado.
- Un ajuste parado a mitad de época se reanuda con el mismo estado elegido, las mismas predicciones y el mismo identificador de ejecución que uno continuo. Una ejecución completa no se repite.
- Ambas entradas se detienen con la protección bloqueada antes de crear salidas.
- El traslado parte del estado del ancla: registra su huella, coincide bit a bit con un recorrido independiente con ese estado y difiere de una inicialización nueva con la misma semilla. Rechaza una ventana no posterior, un ancla de otra vista, parámetros o estado elegido alterados y una salida existente.
- En el traslado, la primera predicción de cada tramo lee un banco vacío aunque el ancla admitió episodios en su evaluación, cada predicción lee exactamente las admisiones anteriores a su instante y todas las etiquetas maduran dentro del tramo.
- Una vista conjunta US+CN produce un recibo por mercado al ajustar y al trasladar, y la última ventana US no predice ninguna fila de 2024.
- La campaña con la sección `episodic_gru`, ejecutores reales en CPU para la candidata y dobles para la GRU de referencia completa las 13 ventanas de US+CN. Cada trabajo de la candidata evalúa las mismas filas que la referencia, cada traslado parte del estado de su ancla, los recibos publicados coinciden con los escritos por la entrada y el manifiesto de fuentes alimenta la comparación sin filas de 2024.
- El planificador solo crea trabajos de la candidata con su sección, con los recuentos y dependencias previstos, y rechaza variantes, semillas, campos, presupuestos, cabezas o brazos distintos de los declarados.
- Las fases registradas por la ventana son exactamente las de `window_phases` con 12 meses, la evaluación observa entradas anteriores a su inicio y `campaign_case` devuelve el caso de búsqueda, el calentamiento y, si la campaña la declara, la parada temprana con la misma métrica.

`test_candidate_run.py` comprueba además que los dos casos de búsqueda y el calentamiento son los de la receta de Titans-MAC, que el caso es obligatorio y que la receta se rechaza si una variante cambia la tasa, si la base la declara, si los casos cambian otros hiperparámetros o si el calentamiento falta o supera 60 meses. [`test_walk_forward_warmup_parity.py`](../../tests/training/test_walk_forward_warmup_parity.py) comprueba en todas las ventanas de los protocolos de la comparación que ninguna fase observa algo anterior al origen ni posterior al final de su tramo, y [`test_joint_stop_trainers.py`](../../tests/training/test_joint_stop_trainers.py) recorre la parada conjunta con este entrenador, el de Titans-MAC y el lector de MARS-TITAN.

El recuento de pruebas, las mutaciones dirigidas y las versiones están en el [recibo técnico](../../reports/engineering/candidate-chronological-trainer-20261009.json), en el [recibo de memoria](../../reports/engineering/candidate-trainer-memory-20261009.json) y en el [recibo de las ventanas](../../reports/engineering/candidate-walk-forward-20261009.json).

## Pendiente

- Perfilar en `cuda:0` el recorrido completo, incluidas las sincronizaciones de las comprobaciones de finitud del nativo y la memoria de los grafos del tramo. La comprobación [`cuda_candidate_run_check.py`](../../tests/training/cuda_candidate_run_check.py), que compara en CPU y `cuda:0` un ajuste completo sin pasos y su validación en FP32 y FP64 con K = 1 y 4, además de una ventana v2 y su traslado, pasó sus seis casos el 9 de octubre ([recibo](../../reports/engineering/cuda-checks-20261009/candidate-trainer-cuda.json)). El 10 de octubre se repitió con el calentamiento de 12 meses de la receta en la ventana v2, que ahora comprueba que ajuste y traslados registran esos meses con las mismas fases en CPU y en `cuda:0`. Pasaron los seis casos ([recibo](../../reports/engineering/cuda-adapter-checks-20261010/candidate-trainer-cuda.json)). La orden, desde `native/`:

  ```bash
  cmake --preset native-candidate-cuda -B ../build/native/candidate-trainer-cuda \
    -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON \
    -DMARS_TITAN_TORCH_ENVIRONMENT=<entorno uv con PyTorch CUDA>
  cmake --build ../build/native/candidate-trainer-cuda -j 2
  cd .. && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
    MARS_TITAN_EPISODIC_NATIVE=$PWD/build/native/candidate-trainer-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so \
    MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT=$PWD/candidate-trainer-cuda.json \
    UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
    uv run --no-sync python -m pytest -q tests/training/cuda_candidate_run_check.py
  ```

  El mismo archivo compara también el ajuste de una ventana v2 y su traslado a la siguiente en CPU y en `cuda:0`, y el traslado en `cuda:0` del estado elegido en CPU, con la misma inicialización de la semilla en ambos dispositivos.
- Calentamiento episódico con etiquetas anteriores al tramo medido, que necesita indexarlas en las observaciones financieras.
- Ventanas sobre la edición real. Las entradas por ventana solo se han comprobado con vistas v2 del corpus técnico, con un activo por mercado.
- Copiar la sección `episodic_gru` a la configuración de A, con la opción de memoria elegida y los límites ampliados.
- Índices de observaciones compartidos. Cada ajuste y cada traslado preparan los suyos aunque solo dependen de la vista y del tramo, así que las semillas repiten ese trabajo. Conviene medir antes su coste con la edición real.
- Presupuesto definitivo y alternativa de reentrenamiento por ventana en [#363](https://github.com/GonxKZ/mars-titan/issues/363), con el caudal medido.
- Elegir la opción de memoria con la [medida en `cuda:0`](#medida-en-cuda0). La medida se repite con el enlace CUDA compilado como arriba, desde la raíz:

  ```bash
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
    MARS_TITAN_EPISODIC_NATIVE=$PWD/build/native/candidate-trainer-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE=cuda:0 \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS=128,256,512 \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT=$PWD/candidate-memory-cuda.json \
    UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
    uv run --no-sync python -m pytest -q tests/training/candidate_memory_check.py
  ```

  La recomputación es la que acota la memoria con cualquier patrón de etiquetas sin cambiar el gradiente, en CPU y en `cuda:0`.
