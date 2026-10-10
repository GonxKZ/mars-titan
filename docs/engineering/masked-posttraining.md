# Postentrenamiento con la edición histórica y matriz de adaptadores

Este documento describe cómo el postentrenamiento lee la edición `historical_masked_2000_v1`, cómo se declara la matriz finita de adaptadores de [#364](https://github.com/GonxKZ/mars-titan/issues/364), con los controles de [#128](https://github.com/GonxKZ/mars-titan/issues/128), cómo se extiende a las familias con entrenador cronológico (Titans-MAC, los lectores episódicos de MARS-TITAN y CM-v1 y la GRU candidata) y cómo se recorre por ventana walk-forward de la campaña con máscaras. Es una preparación técnica. No se ha ajustado ningún brazo ni se ha ejecutado ningún paso de optimizador. El bloqueo de aprendizaje sigue vigente y la reserva de 2024 permanece cerrada.

## Solo datos reales

Los postentrenamientos de las campañas A y B aprenden solo con datos reales de la edición y con la división walk-forward de cada ventana. El postentrenamiento emparejado de #128, que comparaba la condición real con episodios remuestreados (`real_resampled`) y sintéticos (`real_synthetic`), pertenece a un experimento anterior. Se conserva por trazabilidad y queda excluido por construcción de la etapa de adaptadores:

| Garantía | Cómo se cumple |
| --- | --- |
| Entradas | `PairedInputs` solo lee cohortes reales y no acepta episodios. Los episodios añadidos viven en `posttraining/augmented_inputs.py` (`AugmentedInputs`), que solo usa la cola de #128. El orden de visitas de las cohortes reales está en `environments/cohort_order.py` y el aumento lo reutiliza con el mismo generador, así que los lotes reales no cambian |
| Vistas | La etapa solo abre las vistas de la campaña base, que comprueba sus huellas, la política con máscaras, el protocolo y la supervisión de cada mercado. Su identidad guarda `training_data: real_walk_forward_only` y la edición verificada de cada mercado en cada ámbito |
| Declaración | `load_stage` rechaza una etapa o una matriz que declare condiciones, aumento, remuestreo, episodios o mundos, o los valores `real_resampled` y `real_synthetic`, y una campaña sin la política con máscaras. Cada caso neuronal debe tener la condición real |
| Código alcanzable | Una prueba recorre todas las importaciones de `scripts/run_masked_campaign.py`, también las que están dentro de funciones o se hacen por nombre, y comprueba que ninguna llega a `posttraining/queue.py`, `posttraining/preparation.py`, `posttraining/augmented_inputs.py`, `episodes/augmentation.py`, `episodes/worlds.py` ni `episodes/windows.py`. Otra ejecuta `posttraining check` de A y B en otro proceso con esos módulos bloqueados. Los recorridos completos de la etapa en las pruebas sustituyen esos módulos por centinelas que fallan si algo los pide |

La caché de predicciones del padre (`episodes/parents.py`) sí se usa, porque guarda las predicciones del padre sobre cohortes reales. El código de #128 conserva sus pruebas para poder reproducir el experimento anterior, pero no forma parte de ninguna etapa de la campaña.

## Política de entradas

La política se declara en cada lector y se comprueba contra lo confirmado en disco. Sin declararla se conserva la ruta estricta.

| Componente | Cambio con la política histórica |
| --- | --- |
| `prepare_causal_corpus` | Lee `CorpusDataset` con la política, añade una columna `presence` con cinco booleanos por fila y registra `input_policy` y `mask_contract` en el recibo y en su identidad. La huella de `data/input_policy.py` solo entra en este caso. |
| `ParquetCohortSource` | Exige `input_policy` igual a la del recibo, también dentro de su identidad. Devuelve `presence` en el mismo orden de activos que `read_cohort`, comprueba que precios y gráficos están presentes y que cada bloque ausente solo contiene ceros. |
| `PairedInputs` | Solo admite la condición `real`, porque los episodios sintéticos no tienen bits ni causas de ausencia. Las características del adaptador lineal son modalidades, cinco bits y la predicción del padre, en ese orden. |
| `ParentCache` | La clave incluye los bits ordenados. Un bloque ausente y un valor observado igual a cero no comparten predicción. |
| `heldout.run_evaluation` | Lee la política del corpus ordenado confirmado y abre `CorpusDataset` con ella para calibración y evaluación. |
| Cola (`posttraining.queue`) | El diseño puede declarar `input_policy: historical_masked_2000_v1`, solo con la condición real. La política llega al corpus ordenado y a sus lectores. |

Un lector estricto que recibe un corpus con máscaras falla por tres vías independientes: la política declarada en el recibo, la de la supervisión de origen y la presencia de la columna `presence`. Una prueba usa un recibo de dos particiones sin vista temporal, de modo que solo la declaración del recibo separa ambas ediciones.

## Padres

`load_parent` lee `mask_fusion` de la identidad del padre y construye `MultimodalReference` con la misma fusión. La fusión con presencia sin política histórica, o la política sin esa fusión, se rechazan antes de cargar pesos. Se admite también el Transformer compacto como padre neuronal, con la huella de `models/baselines/transformer.py` en su contrato de inferencia.

Los padres de las referencias con máscaras usan `heldout_full_train_sessions_v1`. El recibo debe tener predicciones completas de validación, calibración y evaluación y el resumen por sesión del ajuste. La comprobación de pesos finitos ignora ahora el estado adicional del Transformer, que es un diccionario de contrato. Antes fallaba al cargar cualquier padre Transformer.

Los padres tabulares reciben los cinco bits después de las modalidades, con el mismo orden de `training.tabular_corpus`. Ridge exige además `feature_order` con `presence` al final y 1.719 columnas con las formas de la edición. XGBoost declara la política en su identidad.

Un padre con la [cabeza de cuantiles](quantile-head.md) se reconstruye con `head: quantile_head_v1` y exige `output_head` con el contrato de la cabeza y la huella de `models/quantile_head.py`. Su predicción puntual es la mediana, que alimenta la caché del padre, la corrección lineal de los diseños de #128 y el control del padre congelado. La continuación completa y los adaptadores de un padre de cuantiles optimizan su propia pinball con la [matriz de versión 2](#objetivo-por-cabeza-de-salida). `run_case` rechaza antes de crear el optimizador un objetivo puntual sobre un padre de cuantiles y una pinball sobre un padre escalar.

El contrato del padre solo añade `input_policy`, `mask_contract`, `mask_fusion`, `prediction_retention` y `output_head` cuando el padre los declara. `prepare_parent_cache` rechaza los padres con máscaras. Esa caché alinea predicciones exportadas tras seleccionar el checkpoint, que no son pronósticos emitidos en cada fecha. En esta edición el postentrenamiento recalcula el padre congelado sobre las entradas efectivas. El checkpoint del padre se eligió con la validación de su ventana, así que la validación tampoco es independiente de esa selección. Calibración y evaluación siguen siendo el contraste reservado.

## Normalizador por ventana

Con la edición histórica, `fit_normalization` no guarda la huella del manifiesto ordenado, que también describe validación. Guarda la ventana de ajuste:

| Campo | Contenido |
| --- | --- |
| `train_partition_sha256` | Huella del Parquet de ajuste de la ventana |
| `train_rows` y `train_cohorts` | Filas y sesiones de ajuste |
| `train_bounds` | Inicio, fin y corte del ajuste por mercado, en microsegundos |

`run_case` exige esa ventana, la política y el padre, y rechaza un normalizador de otra ventana, uno con el formato estricto o uno de otra política. La ruta estricta conserva `train_sha256` y el mismo diccionario byte a byte.

La prueba de no fuga prepara dos corpus de la misma ventana. En el segundo cambian los gráficos, un indicador macro observado y los objetivos de todas las filas posteriores al ajuste. El Parquet de validación cambia y el de ajuste no. Los dos normalizadores son iguales como diccionarios completos y la rejilla de 21 acciones conserva sus valores. Otra prueba sustituye la fuente de validación por una que falla al leer y comprueba que el ajuste del normalizador no la abre. `training.predictive_inputs.fit_standardizer` tiene pruebas equivalentes sobre la vista temporal estricta, con una caché técnica en lugar de un padre Ridge. Esa ruta sigue siendo estricta y rechaza la edición con máscaras.

## Sesiones de hasta 8.192 activos

La sesión más poblada medida en [#378](https://github.com/GonxKZ/mars-titan/pull/378) tiene 4.200 activos US. El corpus ordenado, `FrozenParent.predict` y `ParentPredictions.values` aceptan ahora hasta `MAX_COHORT_ASSETS = 8192` filas por sesión, el mismo límite del entorno. El padre sigue recorriendo bloques de 256 filas.

Los presupuestos siguen siendo explícitos. Una sesión completa con las 1.714 columnas `float32` de la edición ocupa unos 56 MB y cabe en los 64 MiB de bloque y de entradas del padre. Con columnas más anchas `shapes_contract` falla en lugar de recortar activos. Una medición local en CPU, de una sola ejecución con valores sintéticos de 8.192 filas y esas columnas, leyó la sesión y recorrió los lotes del adaptador en 2,3 s. El Parquet ocupaba 67,8 MB, el pico de `tracemalloc` fue de 130 MB y el RSS máximo del proceso creció unos 212 MB. No es una medición de la edición real ni de CUDA.

## Matriz de adaptadores

[La matriz](../../configs/posttraining/adapter-matrix-v1.json) se lee con `posttraining.adapter_matrix.read_matrix`, que rechaza cualquier diseño abierto o desigual. Declara tres puntos de inserción:

| Punto | Tensores | Forma | Estados que invalida |
| --- | --- | --- | --- |
| Cabeza | `head.weight` y `head.bias` | Corrección completa inicializada a cero | Predicciones del padre en caché |
| Lectura | Bloque de consulta de `in_proj_weight` y `out_proj.weight` de cada capa de atención | Bajo rango r = 4, alpha = 4 | Predicciones y salidas de atención |
| Fusión | Primera capa de `fusion` | Bajo rango r = 4, alpha = 4 | Predicciones y representaciones fusionadas |

El bajo rango sigue la factorización de [LoRA (Hu et al., 2022)](https://openreview.net/forum?id=nZeVKeeFYf9): W' = W + (alpha / r) U V, con U a cero y V inicializada con un generador propio. La cabeza usa una corrección completa porque su matriz es de 1 × D y el bajo rango no reduce parámetros. En la lectura solo cambia el bloque de consulta de la proyección empaquetada. Claves y valores quedan congelados, que es el contraste de consulta y salida frente a claves estables propuesto en la [integración](../research/system-integration.md).

Los brazos son las siete combinaciones de uno, dos y tres puntos y un control de rango, `fusion_full_rank`, con la fusión ajustada por una corrección completa. RNN, LSTM, GRU y DLinear no tienen lectura, así que reciben cuatro brazos. El Transformer recibe los ocho. Los brazos inaplicables no se ejecutan ni se cuentan como resultados nulos.

Cada semilla añade tres controles comunes: el padre congelado sin actualizaciones, la corrección lineal residual inicializada a cero y la continuación supervisada de todos los parámetros. En la versión 1, adaptadores y continuación usan `neural_mae` y la corrección lineal usa `mae` sobre su centro continuo. Todos comparten semillas 42, 43 y 44, cinco épocas, lote 256, AdamW con tasa 0,0001 y `weight_decay` 0,01, recorte de norma 1 y la selección de versión 2 sin paciencia y con mejora mínima 0,00001. Con la misma población, cada brazo y cada control aplican `5 × Σ⌈n_c / 256⌉` actualizaciones, donde n_c es el tamaño de cada sesión. La selección conserva el mejor estado de las cinco evaluaciones o el padre de la época cero.

La métrica de selección es la del contrato actual del postentrenamiento, el MAE por sesión de la mediana de la rejilla. Cada recibo conserva también los errores del centro continuo y del padre. En la época cero el centro de todos los adaptadores y de la corrección lineal coincide exactamente con la salida continua del padre, pero la mediana de 21 acciones no tiene por qué hacerlo. Si el contraste entre brazos debe usar el centro continuo, hará falta una nueva versión de la selección declarada antes de ejecutar.

### Objetivo por cabeza de salida

Las referencias neuronales de la campaña con máscaras usan la [cabeza de cuantiles](quantile-head.md), así que sus padres no tienen un centro escalar que ajustar con `neural_mae`. La [versión 2 de la matriz](../../configs/posttraining/adapter-matrix-v2.json) conserva puntos, brazos, presupuesto y semillas de la versión 1 y declara el objetivo de cada caso según la cabeza del padre. La versión 1 no cambia y solo admite padres escalares. `adapter_matrix.plan` lee la cabeza del modelo cargado y aplica sus objetivos.

| Cabeza del padre | Adaptadores | Continuación completa | Corrección lineal residual |
| --- | --- | --- | --- |
| Escalar | `neural_mae` | `neural_mae` | `mae` |
| `quantile_head_v1` | `neural_pinball` | `neural_pinball` | Excluida, con motivo declarado |

`neural_pinball` es la pinball media de los cinco niveles por fila (`models/quantile_head.pinball_loss`), promediada después entre filas. Es la misma pérdida con la que se ajustó el padre. La matriz exige el mismo objetivo para adaptadores y continuación, para que su contraste no mezcle el cambio de parámetros con el de pérdida. El orden de los niveles no depende del objetivo: la cabeza construye la mediana libre y los demás niveles con incrementos softplus, de modo que cualquier valor de sus pesos, incluida la corrección del brazo `head`, produce cinco niveles no decrecientes.

La corrección lineal residual de #128 corrige la predicción escalar que guarda la caché del padre, que en un padre de cuantiles es solo la mediana. No produce cinco niveles ni puede optimizar su pinball, y la corrección por nivel inicializada a cero ya es el brazo `head`. Por eso la versión 2 la excluye para esta cabeza con un motivo de al menos 20 caracteres, que `excluded_controls` copia en el plan. La matriz solo admite la exclusión cuando la cabeza no tiene ningún objetivo lineal definido. El control excluido no se ejecuta ni se cuenta como resultado nulo.

Un caso pinball añade a la identidad de su ejecución el campo `objective`, con la pérdida, la cabeza, los niveles, la reducción y el punto de selección, y la huella de `models/quantile_head.py`. Los casos escalares no añaden el campo y conservan su identidad anterior. Las estadísticas del ajuste usan la mediana.

La selección de un caso de cuantiles usa el MAE por sesión de la mediana de la cabeza (`primary: quantile_median`), el punto que publica la campaña. No usa la mediana de la rejilla de 21 acciones, que solo se aplica a un centro escalar. La validación registra también la pinball media y el error del padre con las mismas filas. En la época cero la mediana de todos los brazos y de la continuación coincide con la del padre, así que el padre sigue siendo elegible. Validación, calibración y evaluación escriben `prediction` y `center` iguales a la mediana y las cinco columnas `QUANTILE_COLUMNS`.

### Identidad y parámetros

Cada caso de un brazo lleva `adapter` con la huella de la matriz, la política, el nombre del brazo y la forma resuelta de cada punto. La identidad de la ejecución añade los tensores modificados, sus formas, filas, rango y escala, el recuento de parámetros entrenables y la semilla de V, derivada de la semilla del caso, el brazo y la huella de la matriz. El modelo construido debe tener exactamente ese número de parámetros entrenables. El optimizador solo recibe esos parámetros.

Un checkpoint de otro brazo se rechaza por identidad, aunque sus tensores tengan las mismas dimensiones. La evaluación de calibración y evaluación reconstruye el brazo con sus destinos declarados antes de cargar el estado seleccionado.

`adapter_matrix.plan` registra antes del primer ajuste filas, actualizaciones, parámetros entrenables, bytes de pesos y momentos de AdamW y estados invalidados, y exige el mismo número de actualizaciones en todos los brazos. Con dimensión 64, las formas de la edición y bits de presencia, los parámetros entrenables son:

| Brazo o control | GRU, 1 capa | Transformer, 1 capa | Transformer, 2 capas |
| --- | --- | --- | --- |
| Cabeza | 65 | 65 | 65 |
| Lectura | No aplica | 1.024 | 2.048 |
| Fusión | 1.556 | 1.556 | 1.556 |
| Cabeza, lectura y fusión | No aplica | 2.645 | 3.669 |
| Fusión de rango completo | 20.800 | 20.800 | 20.800 |
| Corrección lineal | 1.721 | 1.721 | 1.721 |
| Continuación completa | 124.033 | 144.385 | 182.017 |

Un recuento menor no se presenta como ahorro de tiempo. Las actualizaciones siguen recorriendo el padre completo en cada lote.

### Familias con entrenador cronológico

La [versión 3 de la matriz](../../configs/posttraining/adapter-matrix-v3.json) conserva puntos, brazos, controles, objetivos, presupuesto y selección de la versión 2 y añade `architectures.chronological`, que `posttraining/chronological_matrix.py` valida y convierte en casos. Un caso cronológico es la continuación completa o un brazo aplicable, con `neural_pinball` y la selección de versión 2. No hay corrección lineal, excluida para la cabeza de cuantiles, ni trabajos del padre congelado.

| Familia | Puntos y tensores | Casos por semilla | Congelado o excluido |
| --- | --- | --- | --- |
| Titans-MAC | Cabeza (`head`, corrección completa), lectura de MAC (`mac.query_projection` y `mac.attention.out_proj`, rango 4) y fusión (`fusion.0`, rango 4) | Ocho brazos y la continuación en `mac_frozen` y `mac_online`. Cuatro y la continuación en `transformer_direct` y `mac_disabled` | `mac.memory` (pesos rápidos y sus proyecciones) y `mac.persistent` |
| MARS-TITAN y CM-v1 | Núcleo `mac_online` con los destinos de Titans-MAC, lectura episódica (`query_projection` y `value_projection` del lector, rango 4) o ambos | Tres brazos y la continuación del lector. Sin banco (M0), solo el núcleo y la continuación | `refinement` y `step_logit` del lector |
| GRU candidata | Cabeza (`head_weight` y `head_bias`) | Un brazo y la continuación | Lectura y fusión, dentro del módulo LibTorch |

La lectura de MAC no existe en `transformer_direct`, que no tiene MAC, ni en `mac_disabled`, que no lee la memoria. Allí la proyección de consulta no interviene en la predicción, y adaptar solo `out_proj` sería otro punto con otros parámetros. La matriz exige un motivo escrito para cada punto que falta en una variante y para cada punto excluido de la GRU candidata. Adaptar su lectura o su fusión exigiría parámetros nuevos dentro de `Candidate::read` y `Candidate::encode_context`, con su archivo, su huella, su enlace en LibTorch, una referencia en Python y compilaciones CPU y CUDA.

Fases, calentamiento, truncamiento, bloques, acumulación, política de memoria, admisión del banco y K se leen de la identidad del ajuste del padre. Solo cambian el optimizador, el presupuesto y la selección de la matriz. Así una regla nueva de un brazo base, por ejemplo su calentamiento, llega igual a sus adaptadores.

- **Titans-MAC.** `ChronologicalTrainer` recibe la declaración del caso en `posttraining` y solo ajusta lo que requiere gradiente: las correcciones o, en la continuación, todo el ajuste externo. Las correcciones forman su propio papel en el optimizador (`adapters`). Los tensores originales, la memoria persistente y los pesos rápidos iniciales quedan congelados y la identidad los enumera. Los pesos rápidos siguen su actualización asociativa en cada observación. Sin postentrenamiento, los papeles y la identidad conservan literalmente su forma anterior, y un predictor con adaptadores sin declaración se rechaza. Con dimensión 64 y los cinco bits de presencia en la entrada de la fusión, la fórmula que comprueba la prueba de recuentos da 325 parámetros en la cabeza, 1.024 en la lectura, 1.556 en la fusión, 2.905 en los tres puntos y 20.800 en la fusión de rango completo.
- **Lectores episódicos.** `ReadoutAdapterTrainer` recorre los mismos instantes, bancos y etiquetas maduras que `ReadoutTrainer`. En el brazo de la lectura el núcleo sigue congelado y se reutiliza su repetición. En el del núcleo la predicción depende de parámetros del núcleo, así que cada tramo se emite con el mismo cálculo diferenciable y al actualizar se repite el núcleo por bloques de flujos desde el estado de cada flujo al empezar el tramo, con BPTT dentro del tramo. El lector se aplica sobre cada bloque repetido con la misma instantánea del banco que vio al emitir. Con la penalización C de CM-v1, el objetivo del núcleo suma sus términos sobre el operador MAC que se ejecuta, que ahora incluye los adaptadores. El banco episódico (M) no se adapta.
- **GRU candidata.** La cabeza adaptada se calcula en Python sobre el estado final del módulo nativo con la misma parametrización ordenada (`ordered_quantiles`). Con la corrección nula reproduce la cabeza nativa bit a bit, y el módulo nativo queda congelado y fuera del grafo. La continuación ajusta todos los parámetros nativos desde el estado elegido.

Las actualizaciones de un caso cronológico las fija el recorrido de su ventana y no sus parámetros, así que coinciden entre los casos de un mismo padre. La etapa lo exige al confirmar cada ajuste.

### Implementación

`ResidualDelta` y `LowRankDelta` son parametrizaciones de `torch.nn.utils.parametrize`. El tensor original queda como parámetro congelado y la red no cambia su código. `adapted_copy` copia el padre, congela todos sus pesos, añade las parametrizaciones y no modifica el padre recibido.

Con correcciones nulas la salida coincide exactamente con la del padre cuando ambos recorren los mismos núcleos. En CPU, el modo de autograd cambia la ruta de LSTM y de la atención también para el propio padre, con diferencias de hasta 1,5e-8 en el fixture. Por eso la paridad exacta se comprueba en inferencia y en entrenamiento por separado, y la validación siempre se calcula en `inference_mode`.

### Variedad de adaptadores

La sección opcional `variety` de la versión 3 añade brazos de un solo punto con formas nuevas en la lectura y la fusión (DoRA, (IA)³ y adaptadores en cuello de botella en paralelo y en serie) y subconjuntos del padre (todos los sesgos, las normalizaciones y la memoria persistente de Titans-MAC). Cada brazo tiene su identidad, el mismo presupuesto y la misma selección, parte exactamente del padre y declara en qué ámbitos se propone para la campaña (referencias, cada variante de Titans-MAC o lectores). En los demás queda como reserva. Las [fuentes, descartes, ecuaciones y comprobaciones](adapter-variety.md) están en su propio documento. La etapa A declara la versión 3, así que programa los brazos propuestos para las referencias y para `titans_mac_online` y sus ajustes pasan de 7.182 a 10.332. B sigue en la versión 2 y no cambia.

## Ejecución de la matriz

`posttraining/matrix_runs.py` reúne las piezas que comparten la cola y la etapa de la campaña. Ninguna decide qué padres o ventanas se recorren.

| Pieza | Función |
| --- | --- |
| `MatrixWindow` | Prepara una vez la lectura de ajuste y validación de una vista con la política de la matriz, el índice de cohortes de la [lectura por bloques](#lectura-por-bloques) o la copia ordenada, la reanuda si existe y abre sus dos lectores. Con `since` el ajuste solo lee las sesiones desde ese instante y la rejilla se ajusta con sus objetivos |
| `MatrixParent` | Carga el padre, abre su caché de predicciones, ajusta el normalizador solo con el tramo de ajuste y registra `plan.json` antes del primer ajuste. `run` llama a `run_case` y exige las actualizaciones del plan |
| `predict_heldout` | Reconstruye el estado seleccionado y escribe calibración y evaluación (y validación, en el walk-forward por etapas) con las filas completas de la vista |
| `release_ordered` | Retira las copias Parquet de ajuste y validación que declara el manifiesto, después de comprobar su huella, y conserva el manifiesto |

### Presupuesto

`plan.json` guarda la huella de la matriz, la cabeza del padre, las actualizaciones por época de su ventana, los controles excluidos y una fila por caso con filas, actualizaciones, parámetros entrenables, bytes de pesos y momentos y estados invalidados. La primera fila es el padre congelado, con cero actualizaciones. `adapter_matrix.plan` exige el mismo número de actualizaciones en todos los brazos y controles del padre, y `MatrixParent.run` rechaza un ajuste completado que aplique otro número. Al reanudar, un plan distinto del registrado detiene la ejecución con «El plan de la matriz de este padre ha cambiado». Como cada ventana tiene otra población, el número cambia entre ventanas, pero no entre los casos de un mismo padre.

### Cola

`run_queue` decide el modo por el `kind` de la configuración. Un diseño de #128 sigue el recorrido anterior, con los ocho objetivos de cada padre y el resumen `paired_posttraining_queue` o `real_continuations_queue`. Una matriz (`posttraining_adapter_matrix`) pasa a `run_matrix_queue`, que escribe el resumen `adapter_matrix_queue`. Los dos modos no comparten salida ni identidad, y `read_design` sigue rechazando una matriz.

En el modo matriz, los padres son los ganadores y finalistas emparejados por semilla de la campaña de referencias de una vista (`matching_parents` con las semillas de la matriz), solo de las familias neuronales. La cola prepara una ventana con la vista de la supervisión y recorre cada padre y semilla con `MatrixParent`. La cabeza de cada padre se lee de su recibo para contar los casos previstos: un padre de cuantiles tiene cinco casos por semilla en RNN, LSTM, GRU y DLinear (cuatro brazos y la continuación) y nueve en el Transformer, y uno escalar añade la corrección lineal. El resumen confirma cada ejecución con la huella de su recibo y sus actualizaciones, junto con el plan de cada padre. Una matriz o un código cambiados, o un caso pendiente al final, detienen la cola. Los puntos de control de recuperación se escriben cada 300 s.

### Lectura por bloques

La copia ordenada de los tramos de ajuste y validación duplicaba en disco cada ventana. En la ventana US+CN más poblada (14,64 millones de filas de ajuste) eran unos 160 GB transitorios, según la medida de [#425](https://github.com/GonxKZ/mars-titan/pull/425). `environments/view_cohorts.py` evita esa copia sin cambiar los lotes:

- `prepare_cohort_index` recorre una vez cada tramo con el mismo lector que alimentaba la copia (`CorpusDataset.batches` con época y semilla cero, cuyo contenido no depende del tamaño de lote) y guarda solo el índice: sesiones y filas por sesión, mercados y límites, las comprobaciones de cada cohorte y la rejilla de acciones, ajustada con los objetivos en el mismo orden que la consulta de la copia (instante y activo). Su identidad incluye la huella de la vista, la política, la supervisión y el código. Un índice de otra vista, política o código se rechaza.
- `ViewCohortSource` entrega las cohortes en el orden de las visitas de cada época, la permutación del generador `[seed, epoch]`. Agrupa visitas consecutivas en bloques que caben en `max_block_bytes`, lee el tramo una vez por bloque y conserva solo las filas de las sesiones del bloque. Un vector nulo o idéntico al primero de su sesión (el macro de una sesión o las noticias ausentes) no se copia y se reconstruye con los mismos bits, también el signo de los ceros. Si un bloque supera el presupuesto mientras se lee, sus últimas sesiones pasan al siguiente, de modo que cada bloque es un prefijo completo de visitas. Con `since` solo quedan las sesiones con decisión desde ese instante, sin cambiar el índice ni su población. Si todas las sesiones de la fuente caben en el presupuesto, se leen una vez y sirven a todas las épocas.

Las pruebas comparan las dos lecturas sobre una vista conjunta US+CN con presencias distintas por mercado. Coinciden bit a bit el índice y la rejilla, cada cohorte con sus bits de presencia en las dos particiones y con varios presupuestos, también con bloques desalojados, los lotes y cursores del adaptador en dos épocas y en validación, la reanudación a mitad de cohorte y la normalización. Una etapa completa con cada lectura aplica los mismos gradientes en cada llamada al optimizador y escribe las mismas predicciones. Al probarlo apareció un fallo anterior: `PairedInputs` no admitía una fuente con dos mercados, que no tiene un límite único de tramo. Ahora cada fila se comprueba con los límites de su mercado.

Medida local en CPU sobre el corpus técnico conjunto, con 16 activos, gráficos distintos por fila y 3.984 filas de ajuste:

| Lectura | Disco tras preparar | Pico de disco | Pico RSS | Lecturas del tramo de ajuste en tres recorridos |
| --- | ---: | ---: | ---: | ---: |
| Copia ordenada | 8,6 MB | 33,6 MB | 955 MB | 3 |
| Bloques de 4 MiB | 37,7 kB | 0,25 MB | 792 MB | 9 |
| Bloques de 32 MiB | 37,7 kB | 0,25 MB | 794 MB | 3 |

El pico de disco de los bloques es la caché del padre, igual en las dos lecturas. Es una sola ejecución de cada caso sobre un corpus pequeño, sin la edición real ni CUDA.

El orden exacto tiene un coste de lectura. Cada bloque es una lectura completa del tramo de ajuste. En la ventana conjunta más poblada, la columna de presencia del tramo de ajuste tiene un 15,0 % de filas con noticias y un 32,6 % con fundamentales, lo que da unos 3.794 bytes por fila en un bloque. Con 3 GiB son 18 lecturas por época si se ajusta todo el tramo, como en el plan anclado de B. En el walk-forward por etapas de A el ajuste solo lee las filas nuevas. Con el caudal del lector medido en `perf/campaign-pipeline` sobre vistas reales (de 17.000 a 25.000 filas por segundo en secuencia y unas 40.000 con hilos), cada lectura de esa ventana tarda entre 6 y 14 minutos. Es una estimación, no una medida de la etapa. Las alternativas (leer en el mismo recorrido todos los ajustes de una semilla y ventana, saltar las filas ajenas al bloque en el lector o declarar otro orden de visitas) están descritas en [#363](https://github.com/GonxKZ/mars-titan/issues/363) y ninguna está aplicada.

## Etapa por ventana de la campaña

`posttraining/campaign_stage.py` recorre la matriz sobre la [campaña con máscaras](../research/training-campaign-2000.md#orquestación-de-los-brazos-con-entrenador) ya confirmada como un walk-forward por etapas, el diseño elegido el 9 de octubre (opción 1 con control). El padre de la ventana k es el estado que la campaña base eligió en k-1 y cada caso lo ajusta solo con las filas de k que ese padre no usó. La etapa se declara en dos configuraciones, una por variante de presupuesto: [A](../../configs/posttraining/historical-masked-adapter-stage-a.json) y [B](../../configs/posttraining/historical-masked-adapter-stage-b.json). Cada una indica su campaña, su matriz, los ámbitos, los brazos, la lectura de las cohortes y los límites de trabajos. A declara la matriz de versión 3 y once brazos base: las cinco redes (RNN, LSTM, GRU, DLinear y Transformer compacto), Ridge, XGBoost y los cuatro brazos de Titans-MAC de la campaña (`titans_transformer_direct`, `titans_mac_disabled`, `titans_mac_frozen` y `titans_mac_online`). Son los once predictores que lee la [etapa de políticas](../../configs/simulation/historical-masked-rl-stage-a.json), así que cada uno publica su cadena. B conserva la versión 2 y las cinco redes. `load_stage` exige que la matriz declare objetivos para `quantile_head_v1`, que comparta política y lote con la campaña, que los ámbitos sigan su orden, que los brazos pertenezcan a la comparación y que las semillas de cada brazo con casos sean las de la matriz. A debe declarar además `chain_rule: chain_validation_score_v1`, `data_policy: real_edition_only` y la lectura por bloques, y su campaña debe reentrenar todas las ventanas. Con la matriz de versión 3 admite también los [brazos cronológicos](#brazos-cronológicos).

| Elemento | Regla |
| --- | --- |
| Padre | Estado elegido del brazo base en la ventana anterior y la misma semilla, según los recibos confirmados de la campaña: el ganador de la búsqueda con la semilla 42 y el finalista con las demás. El informe debe conservar su huella. La primera ventana de cada ámbito no tiene postentrenamiento |
| Ajuste | Solo las filas de `train_k` con decisión desde el final de `cal_{k-1}` hasta el final de `train_k`, con la [disjunción](#filas-nuevas-y-disjunción) comprobada por identidad de fila. La caché y el normalizador son los de ese padre sobre esas filas |
| Selección | Validación de la ventana k, con la selección de versión 2 y el padre elegible en la época cero |
| Predicción | Validación, calibración y evaluación de la vista k con el estado seleccionado, el esquema común de la comparación, la mediana como predicción y los cinco cuantiles |
| Filas | Las de la vista. Calibración y evaluación deben tener la huella de filas y objetivos de la campaña base en esa ventana |
| Brazos | `<brazo base>__frozen_parent` (el padre congelado), `<brazo base>__<punto>` para cada caso de la matriz, por ejemplo `gru__head_fusion`, y `<brazo base>__full_continuation` |
| Reentrenamiento base de k | Es el contraste de la cadena y no compite como candidato |

Cada trabajo confirma `jobs/<ámbito>/<ventana>/<brazo>/<fit|frozen>-s<semilla>/receipt.json` con su identidad (etapa, caso, vista de la ventana y del padre, trabajo base del padre con la huella de su recibo y de su estado elegido y, en un ajuste, la huella de sus filas nuevas y de su prueba), la ejecución, las actualizaciones, la selección, la puntuación de validación recalculada y la declarada, el resumen de las filas nuevas, `labels_used_until` y las predicciones con su huella, filas, huella de filas y objetivos y huellas por mercado. Antes de confirmarlo se comprueba que cada tramo cae en su segmento sin filas de 2024, que tiene las filas de la vista, que los cuantiles son finitos y no decrecientes y que la predicción es exactamente la mediana. Después se escribe el recibo walk-forward de cada mercado en `windows/<ámbito>/<ventana>/<brazo>/seed-<semilla>/<mercado>.json`, con el contrato de [#390](https://github.com/GonxKZ/mars-titan/pull/390), validado con `read_window_receipt`. Su `parent` es el trabajo de la etapa que dejó el estado seleccionado. Su `labels_used_until` es la madurez de la última etiqueta aceptada de ajuste, validación o calibración de la ventana o del padre, una cota conservadora de las etiquetas que llegaron a ese estado y a su selección, siempre anterior a la evaluación. Hasta que `training/label_maturity.py` de [#430](https://github.com/GonxKZ/mars-titan/pull/430) entre en `develop`, la calcula `staged_rows.labels_used_until` con las mismas etiquetas aceptadas.

### Filas nuevas y disjunción

`staged_rows.posttraining_rows` fija el intervalo de decisiones de las filas nuevas: empieza al final de la calibración del padre y termina con el tramo de ajuste de la ventana, que la vista ya purga antes de `val_k`. En la campaña A cada ventana avanza un año y la validación y la calibración ocupan nueve meses, así que las filas nuevas son las de un trimestre (de enero a marzo del año de la evaluación anterior).

`staged_rows.fit_rows_proof` lo comprueba por identidad de fila. Una fila es un mercado, un activo y su fila del archivo de muestras, que las vistas de una misma edición comparten. Por eso exige que cada activo conserve la huella de su archivo de muestras en las dos vistas. La prueba se calcula una vez por ámbito y ventana y se guarda en `windows-data/<ámbito>/<ventana>/fit-rows.json`:

| Campo | Contenido |
| --- | --- |
| `parent_rows` | Filas de ajuste, validación y calibración del padre en su vista |
| `intersection` | Filas nuevas que el padre usó en cada uno de esos tramos. Deben ser cero |
| `rows`, `sha256` | Número de filas nuevas y su huella, con la fórmula del plan de la campaña (`mars-titan-chain-rows-v1`, activo, número de filas y sha256 de sus filas ordenadas) |
| `first_decision`, `last_decision` | Primera y última decisión de las filas nuevas |
| `parent_labels_mature_until` | Madurez de la última etiqueta del padre. Debe ser anterior al inicio de las filas nuevas |
| `labels_mature_until` | Madurez de la última etiqueta de las filas nuevas. Debe ser anterior a la validación |
| `labels_used_until` | Madurez de la última etiqueta de ajuste, validación o calibración del padre o de la ventana |

La prueba falla si la ventana no tiene filas nuevas, si un activo cambia de archivo de muestras, si una fila nueva ya la usó el padre o si una etiqueta madura fuera de su límite. Cada trabajo la vuelve a leer, comprueba que corresponde a sus dos vistas y guarda su huella en la identidad.

Cada familia lee solo esas filas:

- Brazos neuronales. `ViewCohortSource` recibe `since`, el inicio de las filas nuevas, y deja solo las sesiones desde ese instante. El índice y su población no cambian. La rejilla de acciones se ajusta con los objetivos de las filas nuevas. La etapa exige que el lector de ajuste tenga exactamente las filas de la prueba. El padre se carga con el índice de su propia vista, porque su identidad depende de la población con la que se ajustó.
- Titans-MAC y lectores episódicos. La fase de ajuste decide en el intervalo de las filas nuevas y lee antes, sin etiquetas, el mismo calentamiento de 12 meses que los tramos medidos, sin salir del tramo de ajuste de la vista. Validación, calibración y evaluación son las de la ventana k con su calentamiento. El estado se carga como en un traslado. Con M3 se conservan las escalas congeladas del ajuste del brazo padre, y el entrenador comprueba que son esas y que terminan antes de las filas nuevas.
- GRU candidata. No tiene calentamiento: su contexto de 64 sesiones viaja en cada muestra.

### Padre congelado

El trabajo `frozen` de cada brazo base, semilla y ventana k ≥ 1 aplica sin ajuste el estado elegido en k-1 a validación, calibración y evaluación de k. Usa el mismo recorrido que los ajustes: `evaluate_partition` con el modelo del padre en los brazos neuronales y `frozen_titans`, `frozen_readout` o `frozen_candidate` en los cronológicos, que reinician la memoria rápida en cada tramo y leen su calentamiento como un traslado de la campaña base. Así, un caso que no cambia pesos emite exactamente las mismas filas y empata con el padre. Registra cero actualizaciones.

### Cadena trivial de Ridge y XGBoost

Ridge y XGBoost no tienen puntos de adaptación en la matriz, pero las políticas leen su predictor de la cadena igual que el de las demás familias. Su cadena solo tiene el padre congelado (diseño `frozen_parent_only`). En cada ventana k ≥ 1 el trabajo `frozen` aplica el estado elegido por la base en k-1 con `carry_tabular(..., frozen_parent=True)`, el mismo traslado de la campaña, que además predice la validación de k porque la cadena la puntúa. Su recibo de traslado marca `frozen_parent` y el de la etapa tiene la misma forma que el de cualquier padre congelado, sin columnas de cuantiles porque estos modelos solo emiten la predicción puntual. La selección `chain_validation_score_v1` tiene entonces un único candidato y siempre publica ese padre, con los recibos walk-forward por mercado y la madurez de etiquetas de la ventana. Las semillas son las de la campaña (42 en Ridge y 42, 43 y 44 en XGBoost). Solo existe en A, porque B no tiene cadena, y estos brazos no entran en la [comparación de los brazos postentrenados](../research/metrics.md#brazos-postentrenados) porque no tienen continuación ni adaptadores que contrastar.

### Predictor de la cadena

Por ámbito, ventana, brazo base y semilla, el trabajo `<ámbito>/<ventana>/<brazo base>__chain/select-s<semilla>` elige el predictor publicado con la regla `chain_validation_score_v1`:

1. La puntuación de cada candidato es el MAE medio por sesión de la mediana sobre todas las filas de `val_k`, la misma definición que elige el estado de la campaña base. `staged_chain.validation_score` la recalcula sobre las predicciones de validación guardadas, en orden de mercado, activo e instante, y el recibo se rechaza si no coincide con la declarada por el ajuste con tolerancia relativa 1e-6.
2. Compiten el padre congelado, cada caso de adaptadores y la continuación completa. Gana la puntuación menor.
3. El padre congelado solo se sustituye con una mejora estricta. Los empates entre los demás se resuelven por el identificador del trabajo.
4. En la primera ventana el predictor es el estado elegido de la base.

La selección escribe primero los recibos de cada mercado en `windows/<ámbito>/<ventana>/<brazo base>__chain/seed-<semilla>/<mercado>.json`, con el `parent` del trabajo elegido y la huella de su recibo, y al final `selection.json`, de tipo `campaign_chain_selection`, con la campaña, la etapa, la regla, la ventana del padre, los candidatos con su puntuación, el elegido, su estado, las filas nuevas (nulas en la primera ventana y con el padre congelado), la huella de cada recibo de mercado y `labels_used_until`. `staged_chain.read_selection` comprueba todo el conjunto antes de aceptarlo y una reanudación exige la misma selección. Los identificadores, rutas, regla de elección y lectura son los de `training/campaign_chain.py`, que declara el diseño en el plan de la campaña, y `staged_chain` solo añade el brazo del padre congelado y la puntuación de validación. `staged_rows` toma del mismo módulo las filas nuevas y su huella, así que el plan, la etapa y el verificador de disjunción usan una sola definición.

Dos límites de la regla. La igualdad solo es exacta cuando las predicciones son las mismas. Un caso con pesos casi iguales a los del padre puede ganarle por una diferencia del orden del redondeo, y la regla no fija un margen mínimo. Además, la cadena no alimenta a la ventana siguiente: el padre de k+1 sigue siendo el estado elegido por la base en k.

### Brazos cronológicos

Con la matriz de versión 3, un brazo de Titans-MAC, MARS-TITAN, CM-v1 o la GRU candidata entra en la etapa si la campaña base declara su sección. Uno que solo existe en la comparación queda en espera (`awaiting_sections` en `check`) y no genera trabajos. Sus casos salen de `chronological_matrix.cases` con la variante de Titans-MAC del brazo o con su banco, y los ejecutan `posttraining/chronological_windows.py` (Titans-MAC y lectores) y `posttraining/candidate_adapters.py` (GRU candidata):

- El ajuste parte del estado elegido del brazo base en la ventana anterior y la misma semilla, abre la vista de la ventana y escribe validación, calibración y evaluación con el esquema común. No abre la lectura de cohortes de los brazos neuronales.
- El padre congelado escribe cada intento en una carpeta nueva (`attempt-0001`, `attempt-0002` y siguientes, hasta 16) y la etapa reutiliza el último si está completado.
- Un brazo que parte de otro predictor elegido, como MARS-TITAN de `titans_mac_online`, exige también los recibos confirmados de ese padre en la campaña base.
- Todos los ajustes de un mismo padre deben aplicar las mismas actualizaciones, y el recibo se rechaza si no.
- Tras confirmar un trabajo cronológico se borran sus índices de observaciones, que solo reconstruiría otra ejecución del mismo trabajo. El resumen registra los bytes liberados en `released_index_bytes`.

A declara los cuatro brazos de Titans-MAC con la matriz de versión 3. B mantiene la versión 2 y solo las redes, porque no se ejecuta. Los lectores de MARS-TITAN y CM-v1 y la GRU candidata siguen en la declaración preparada de ampliación y no entran en la etapa mientras la campaña A no declare su sección. La medición de caudal todavía no mide los casos de Titans-MAC, así que la estimación de horas los deja en `without_estimate` junto con la cadena trivial de Ridge y XGBoost.

### Walk-forward por etapas en la campaña A v2

La campaña A v2 declara el [walk-forward por etapas](../research/walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2) y cambia tres reglas de la tabla anterior. El padre de la ventana k es el estado elegido de la base en k-1 con el mismo brazo y semilla. El ajuste solo usa las filas del tramo de ajuste de la vista k con decisión entre el final de la calibración del padre y el final de ese tramo (`campaign_chain.posttraining_rows`), y la primera ventana no tiene posentrenamiento. Además del padre congelado y de la matriz, entra como candidata la continuación completa del padre con esas filas. El predictor de la cadena se elige con `chain_validation_score_v1` y se publica en `windows/<ámbito>/<ventana>/<brazo base>__chain/seed-<semilla>/`, con un recibo #390 por mercado y `selection.json` como marca de confirmación.

La etapa actual todavía sigue la tabla anterior. Por eso el calendario de la campaña v2 rechaza su configuración, con el motivo de la primera dependencia que no cumple, hasta que adopte el contrato. `run_masked_campaign.py disjunction` comprueba las filas nuevas, la última etiqueta de cada recibo de la cadena y las huellas de su selección.

### Variante B

B conserva su configuración y su plan anclado: en las ventanas reentrenadas se ajusta y en las trasladadas el trabajo `carry` aplicaría el estado del ancla. Por la decisión del 9 de octubre no se ejecuta. `check` la cuenta y la marca como no ejecutable con su motivo, `run_stage` la rechaza antes de leer nada y su configuración no puede declarar la regla de la cadena.

### Recuento

Por ventana y semilla hay 82 casos en A: 51 en las redes (nueve en cada familia recurrente y en DLinear y quince en el Transformer) y 31 en Titans-MAC (cinco en `transformer_direct` y en `mac_disabled`, nueve en `mac_frozen` y doce en `mac_online`). De ellos, 25 son brazos de la [variedad](adapter-variety.md): cuatro en cada familia recurrente y en DLinear, seis en el Transformer y tres en `mac_online`. Con tres semillas son 246 por ventana. A tiene 45 ventanas, 42 con postentrenamiento, y 31 padres congelados por ventana: tres semillas en cada uno de los nueve brazos con casos, la semilla 42 de Ridge y las tres de XGBoost. B solo cuenta los 29 casos de las redes de la versión 2.

| Variante | Ventanas con ajuste | Ajustes | Padres congelados | Traslados | Selecciones de la cadena | Se ejecuta |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 42 | 10.332 | 1.302 | 0 | 1.395 | Sí |
| B | 17 | 1.479 | 0 | 2.436 | 0 | No |

Los límites de cada configuración (ajustes y predicciones) son iguales a su plan, como en la campaña base. `check` los comprueba sin leer datos. En la estimación de horas de `training/campaign_throughput.py`, las filas nuevas de cada ventana se aproximan con los recuentos como el ajuste de k menos ajuste, validación y calibración de k-1. Es una cota algo mayor, porque las filas que la purga quitó en las fronteras de k-1 sí están en el ajuste de k.

### Ejecución, recuperación y disco

`run_stage` llama a `require_learning_allowed` antes de abrir fuentes y otra vez antes de cada trabajo pendiente. Lee el estado confirmado de la campaña base con su misma identidad (configuración, vistas y código) y exige que estén confirmados todos los trabajos base de los ámbitos y brazos de la etapa. La salida debe quedar fuera de las vistas, de la salida de la campaña y de `dataset/`, y guarda su identidad en `stage.json`: etapa, diseño, regla de la cadena, política de datos, campaña, matriz, política de entradas, `training_data`, variante, edición de cada mercado, huellas de las vistas y huellas del código. Una salida de otra etapa, otra campaña u otro código se rechaza.

Los trabajos se recorren ventana a ventana: en cada una, el padre congelado y los ajustes de cada padre y después la selección de cada cadena, con una sola ventana, un padre y una vista abiertos. Un trabajo con recibo e identidad iguales no se repite, tras comprobar las huellas de sus artefactos. Uno pendiente se reanuda desde su punto de control con la frecuencia de la campaña. Una parada deja el resumen en `paused`, la protección en `blocked` y un error en `failed`. El resumen registra trabajos previstos y completados (ajustes, padres congelados y selecciones), las actualizaciones de cada trabajo, el elegido de cada cadena y el plan de cada padre.

La declaración `cohort_reading` fija cómo leen los brazos neuronales:

| Lectura | Declaración | Disco por ventana |
| --- | --- | --- |
| Por bloques | `{"source": "view_blocks", "max_block_bytes": n}`, con n entre 256 MiB y 16 GiB | Índice de cohortes en `windows-data/<ámbito>/<ventana>/cohorts/` |
| Copia ordenada | `{"source": "ordered_corpus", "retention": "keep"}` o `"release_after_window_fits"` | Parquet de ajuste y validación. Solo la admite B |

Las dos configuraciones del repositorio leen por bloques con 3 GiB. Si todas las sesiones de una fuente caben en ese presupuesto, la primera lectura las conserva y las épocas siguientes no vuelven a leer el tramo. El intento se hace una vez y, si falla, la fuente sigue por bloques. Con unos 3.794 bytes por fila, la estimación de la ventana conjunta más poblada, caben unas 850.000 filas. Si las filas nuevas o la validación de una ventana no caben, su fuente sigue por bloques. No se ha medido con la edición real cuántas caben. Cada ventana guarda además la prueba de filas nuevas y, por brazo base y semilla, la caché del padre, el normalizador y el plan. `training/storage_budget.py` cuenta las tres particiones de predicción de cada trabajo y solo cuenta la copia ordenada y su preparación cuando la etapa la declara.

```bash
uv run --no-sync python scripts/run_masked_campaign.py posttraining check \
  --stage configs/posttraining/historical-masked-adapter-stage-a.json
uv run --no-sync python scripts/run_masked_campaign.py posttraining run \
  --stage <configuración> --views US=<vistas>/US --views CN=<vistas>/CN \
  --views US+CN=<vistas>/US+CN --campaign-output <campaña> --output <etapa>
```

La etapa está registrada en `LATER_STAGES` de `training/campaign_plan.py` con la matriz de versión 3 de A, la configuración de cada variante, su punto de entrada (`campaign_stage:run_stage`) y ninguna tarea pendiente, y `check` de la campaña la informa así. La orden única de la campaña la ejecuta con el subcomando `posttraining`, que delega en `campaign_stage.main`. La orden propia del módulo sigue disponible con los mismos argumentos. La [medición de caudal](../research/training-campaign-2000.md#medición-de-caudal) de la campaña mide también sus casos y estima sus horas.

## Protección del aprendizaje

`run_case`, `run_queue` en sus dos modos, `run_stage`, `run_completion` y las etapas tabular y de postentrenamiento de la compleción llaman a `require_learning_allowed()` de `training/learning_hold.py` antes de abrir fuentes o crear salidas. Si la protección local está vigente fallan con `LearningHoldError` y el mensaje «Bloqueo de aprendizaje vigente». La evaluación congelada no ajusta parámetros y no cambia. El inventario común de puntos de entrada protegidos está en [la guía de pruebas](../../tests/README.md#protección-del-aprendizaje).

Las pruebas del bloqueo declaran una protección temporal en `tmp_path` mediante `MARS_TITAN_TRAINING_HOLD`. Para que el resto de pruebas del paquete lleguen hasta el paso del optimizador como antes, `tests/posttraining/conftest.py` sustituye la protección que leen los puntos de entrada por una que los admite mediante el fixture común `learning_doubles`. Solo lo hace cuando la protección real está vigente, es decir, cuando `tests/conftest.py` ya ha registrado el gancho que omite cualquier paso de un optimizador de PyTorch.

## Comprobaciones

Todas se ejecutaron en CPU con `CUDA_VISIBLE_DEVICES=-1`, sin pasos de optimizador. Las pruebas del recorrido completo sustituyen AdamW por un optimizador que no hereda de `torch.optim.Optimizer`, registra parámetros y gradientes y exige que los pesos no cambien.

### Lectura con máscaras y matriz

- `tests/models/test_predictive_adapters.py`: paridad exacta de las cinco familias con fusión estricta y con presencia, gradiente solo en los adaptadores, padre intacto, recuento de parámetros, filas de consulta, semilla aislada del RNG global y recarga solo en el mismo brazo.
- `tests/posttraining/test_masked_corpus.py` y `test_masked_inputs.py`: columna de presencia, lectores con política declarada, relleno nulo, características, condición real, normalizador por ventana y no fuga.
- `tests/posttraining/test_masked_parents.py`: padres GRU, DLinear y Transformer con fusión de presencia, padre de cuantiles con su mediana, rechazo de políticas, retenciones y contratos de salida incompatibles, bits tabulares y claves de la caché.
- `tests/posttraining/test_masked_run.py`: brazos y controles hasta el paso del optimizador, igualdad de actualizaciones, pausa y reanudación, rechazo del checkpoint de otro brazo, evaluación reservada con bits y cola con la política declarada.
- `tests/posttraining/test_adapter_matrix.py`: configuración del repositorio, plan, destinos y 17 matrices inválidas.
- `tests/posttraining/test_session_capacity.py` y `tests/training/test_predictive_windows.py`: sesión de 8.192 activos y normalizador por ventana de la ruta estricta.
- `tests/posttraining/test_learning_hold_entrypoints.py`: rechazo de los cinco puntos de entrada antes de abrir fuentes o crear salidas, protección permisiva o ausente y protección ambigua.

Treinta y ocho mutaciones dirigidas sobre adaptadores, lectores, padres, cabeza de cuantiles, normalizador, ejecución, matriz, caché, cola y protección del aprendizaje hacen fallar al menos una prueba cada una. Dos mutantes iniciales sobrevivieron. Uno era equivalente, porque desplazaba el bloque de filas cuando este empieza en la fila cero. El otro retiraba la comprobación de política del lector, que la supervisión de origen también rechazaba. Se sustituyó el primero y se añadió una prueba sin vista temporal para el segundo. La paridad estricta se contrastó fuera de la suite ejecutando el mismo guion con `develop` y con esta rama. Coinciden los Parquet ordenados, la rejilla, las lecturas de cohortes, los lotes del adaptador, el normalizador, la identidad de ejecución, la evaluación reservada, el padre cargado y los casos de los diseños existentes. Solo cambian las huellas de código y las que dependen de ellas.

Las pruebas de los módulos relacionados (postentrenamiento, entornos, episodios, referencias, campañas, evaluación y observatorio) dan los mismos 38 fallos en `develop` y en esta rama. Todos se deben a CUDA no disponible o a `CUBLAS_WORKSPACE_CONFIG` sin declarar.

Quedan sin ejecutar las pruebas que aplican pasos reales de AdamW, que la guarda de `tests/conftest.py` omite, y las que ajustan Ridge: `tests/training/test_predictive_parents.py`, `test_predictive_inputs.py`, `test_predictive_run.py` y `test_predictive_study.py`.

### Objetivo pinball, cola y etapa por ventana

Los recuentos y recorridos de la etapa de este apartado corresponden a su diseño anterior, con el padre en la misma ventana y traslados en B. Los del diseño actual están en [walk-forward por etapas](#walk-forward-por-etapas).

- `tests/posttraining/test_quantile_adaptation.py` (17 pruebas): la pinball solo llega a los adaptadores y deja intacto el padre, la continuación recibe gradiente en todos los pesos, el orden de los cuantiles se conserva con pesos aleatorios, cada caso de cuantiles llega hasta el paso del optimizador con las actualizaciones previstas, se selecciona con la mediana y resume el ajuste con ella, un objetivo que no corresponde a la cabeza se rechaza antes de crear el optimizador y un padre escalar conserva su objetivo e identidad.
- `tests/posttraining/test_adapter_matrix.py` (37): además de lo anterior, la versión 2, su plan con padres de cuantiles y diez disposiciones de objetivos inválidas.
- `tests/posttraining/test_matrix_queue.py` (3): el modo matriz recorre cada caso por padre y semilla, se pausa y se reanuda sin repetir pasos, aplica las mismas actualizaciones en todos los brazos y controles, detecta un plan alterado, no repite casos confirmados, mantiene separado el modo de #128 y se detiene con la protección antes de leer.
- `tests/posttraining/test_adapter_campaign_stage.py` (31 con las de la lectura por bloques): recuentos de A y B con las configuraciones del repositorio, traslados desde el ancla, diez declaraciones inválidas y el recorrido completo sobre una campaña base reducida con padres de cuantiles reales y pesos iniciales (dos ventanas US, un brazo GRU y la semilla 42). En A comprueba diez ajustes con las mismas actualizaciones por ventana, predicciones idénticas a las del padre elegido, recibos walk-forward válidos y copias retiradas. Con la lectura por bloques no hay copia ordenada y las dos lecturas aplican los mismos gradientes y escriben las mismas predicciones. En B comprueba cinco ajustes en el ancla y cinco traslados sin optimizador. También pausa y reanudación, la protección antes de leer, antes de un ajuste y antes de un traslado, y el rechazo de filas de 2024, objetivos distintos, cuantiles desordenados, una predicción distinta de la mediana, un trabajo base sin confirmar y un traslado que no parte del padre del ancla.

Treinta y siete mutaciones dirigidas sobre el objetivo pinball, la evaluación de cuantiles, la evaluación reservada, la matriz de versión 2, la ejecución por padre, la cola y la etapa hacen fallar al menos una prueba cada una. En la primera pasada sobrevivieron cinco: retirar la protección por trabajo (`run_case` la comprueba también, pero un traslado no pasa por él), la comprobación de la mediana publicada, la de los objetivos de cuantiles al cargar la etapa, admitir `neural_mae` para padres de cuantiles y resumir el ajuste con los cinco niveles en lugar de la mediana. Se añadió una prueba para cada una y las cinco fallan ahora.

La paridad con `develop` (69bc59c1) se contrastó fuera de la suite con el mismo guion en los dos árboles y el optimizador registrador. Recorre los seis objetivos lineales y las dos continuaciones neuronales de la ruta estricta de #128, nueve casos de la matriz de versión 1 con máscaras sobre GRU, DLinear y Transformer, su calibración y evaluación, la evaluación congelada de cada padre y los diseños existentes de la cola. Coinciden las 639 llamadas al optimizador y la huella de todos sus gradientes, los Parquet de validación, calibración y evaluación, las métricas, los casos y los recibos. Solo cambian las huellas de código, los checkpoints que las contienen y la huella del manifiesto ordenado, que incluye tiempos de ejecución y también cambia entre dos ejecuciones de la misma rama.

Después de rebasar sobre `develop` (6119a7ea), 39 archivos de pruebas de postentrenamiento, campaña, referencias con cuantiles, recibos walk-forward y comparación dan 585 pruebas superadas, 55 omitidas y ninguna fallida, ejecutados uno a uno. Las omisiones son 42 por el bloqueo (pasos reales de AdamW y campañas de referencias) y 13 por la ausencia de CUDA. `ruff check`, `ruff format --check` y `scripts/check_repository.py` no dan errores.

### Familias cronológicas, lectura por bloques y datos reales

Los recuentos y recorridos de la etapa de este apartado corresponden a su diseño anterior, con el padre en la misma ventana y traslados en B. Los del diseño actual están en [walk-forward por etapas](#walk-forward-por-etapas).

- `tests/posttraining/test_adapter_matrix.py` (54): además de lo anterior, la versión 3, sus casos cronológicos con receta y semillas de componente fijas y las declaraciones abiertas o que tocan la memoria.
- `tests/posttraining/test_titans_adapters.py` (14): con correcciones nulas, cada brazo de las cuatro variantes emite los mismos bits que el padre congelado, los recuentos coinciden con los tensores declarados, el gradiente solo llega a los adaptadores y la memoria queda congelada, los brazos de un padre comparten actualizaciones, etiquetas y predicciones, el gradiente de cada corrección es la regla de la cadena aplicada al de la continuación completa, la reanudación reproduce el recorrido continuo y sin postentrenamiento los papeles y la identidad no cambian.
- `tests/posttraining/test_readout_adapters.py` (12): brazos de MARS-TITAN y CM-v1 con los mismos eventos, actualizaciones y predicciones emitidas, gradiente solo en las correcciones declaradas, regla de la cadena de la lectura, gradiente del núcleo independiente del tamaño de bloque de flujos, núcleo sin banco, penalización C sobre el operador adaptado, reanudación y rechazos.
- `tests/posttraining/test_candidate_adapters.py` (5): la cabeza en Python reproduce bit a bit la cabeza nativa, el brazo emite las filas del padre en los dos mercados, solo se mueven las correcciones de la cabeza, el traslado coincide con el del padre y se rechazan casos de otra matriz o semilla.
- `tests/posttraining/test_readout_windows.py` (3): sobre una campaña B reducida con `titans_mac_online` y `mars_titan_m1`, los cuatro casos de M1 ajustan y trasladan con las filas y los bits del brazo base y las mismas actualizaciones.
- `tests/posttraining/test_chronological_stage.py` (5): la etapa planifica, ajusta y traslada `titans_mac_online` con la continuación y la cabeza sobre la campaña B reducida de Titans-MAC (14 ajustes y 24 traslados), reproduce las filas de la base, libera los índices y no repite trabajos al reanudar.
- `tests/environments/test_view_cohorts.py` (17): índice, rejilla, cohortes, lotes, cursores, reanudación y normalización iguales bit a bit a los de la copia ordenada, con desalojo, y rechazos de índices ajenos.
- `tests/posttraining/test_real_data_only.py` (17): las importaciones del programa de la campaña, la ejecución de `posttraining check` con los módulos de #128 bloqueados, siete declaraciones ajenas a los datos reales, la campaña sin máscaras, la edición de cada mercado en la identidad, entradas reales sin episodios y el orden de las cohortes reales igual al anterior.

Las pruebas de `tests/posttraining/test_adapter_campaign_stage.py` y `test_chronological_stage.py` recorren la etapa con los módulos de #128 sustituidos por centinelas.

### Walk-forward por etapas

Se ejecutaron en CPU, sin GPU visible y sin pasos de optimizador. Las que recorren ajustes sustituyen AdamW por el registrador que exige pesos sin cambios.

- `tests/posttraining/test_staged_rows.py` (8): sobre las dos ventanas US de la campaña reducida, el intervalo de filas nuevas (del 1 de enero al 1 de abril de 2022) y la prueba de disjunción frente a una lectura independiente de las etiquetas de las dos vistas: recuentos del padre, filas nuevas, primera y última decisión, huella con la fórmula del plan y madurez de las etiquetas. Tres inicios desplazados hacia los tramos del padre se rechazan por intersección y, sin esa comprobación, por la madurez de sus etiquetas. También se rechazan las vistas intercambiadas y un archivo de muestras distinto, la huella no depende del orden y `MatrixWindow` ajusta la rejilla solo con las filas nuevas.
- `tests/posttraining/test_staged_chain.py` (15): la puntuación es la media de los MAE por sesión, no depende del orden en que se escribieron las filas y suma cada sesión en un orden fijo. La regla solo sustituye al padre congelado con una mejora estricta, desempata por identificador y rechaza siete conjuntos de candidatos inválidos. Se comprueban además identificadores, rutas, el orden de las ventanas y los padres de cada semilla.
- `tests/environments/test_view_cohorts.py` (26): además de lo anterior, una fuente que cabe en el presupuesto se lee una vez en tres épocas con los mismos bits, una que no cabe sigue por bloques sin reintentar, `since` conserva las sesiones posteriores con los mismos bits con dos presupuestos y rechaza instantes inválidos, y `PairedInputs` identifica el ajuste limitado sin cambiar la validación ni la población.
- `tests/posttraining/test_adapter_campaign_stage.py` (40): recuentos de A y B, B no ejecutable y sin cadena, dependencias de cada ajuste en los padres de k-1, orden ventana a ventana y declaraciones inválidas de A. El recorrido completo usa la campaña A reducida de dos ventanas US con un brazo GRU. Predice fold-001 con el padre congelado y ajusta cinco casos desde el estado elegido en fold-000, con el normalizador limitado a las filas nuevas y las mismas actualizaciones. Las predicciones son idénticas a las del padre congelado, los recibos walk-forward llevan la última etiqueta usada y la cadena conserva al padre congelado. Una mejora estricta de 1e-9 elige el adaptador, una puntuación recalculada distinta de la declarada se rechaza, una selección, un recibo o una prueba alterados detienen la siguiente ejecución, B se rechaza antes de leer, una pausa se reanuda sin repetir actualizaciones y la protección se comprueba antes de leer y antes de cada trabajo.
- `tests/posttraining/test_chronological_stage.py` (7): campaña A reducida de dos ventanas US con `titans_mac_online` ajustado con los ejecutores reales. La etapa predice fold-001 con el padre congelado y ajusta la continuación y la cabeza solo con las filas nuevas, con decisiones desde el 1 de enero de 2022 y el calentamiento anterior sin etiquetas. Sin cambios de pesos, los dos ajustes emiten exactamente las filas del padre congelado en validación, calibración y evaluación, la cadena lo conserva, los índices se liberan y la reanudación no crea optimizadores.
- `tests/posttraining/test_readout_windows.py` (5) y `test_candidate_adapters.py` (6), con el enlace nativo: los casos de M1 y la cabeza de la GRU candidata ajustados con las filas nuevas de la ventana siguiente emiten las filas de su padre congelado, y este reproduce en calibración y evaluación el traslado de la campaña base.
- `tests/posttraining/test_readout_adapters.py` (13): además, la regla de las escalas M3 por etapas (las del brazo padre, anteriores a las filas nuevas) y la del ajuste base.
- `tests/posttraining/test_heldout.py`: la evaluación congelada admite validación y sigue rechazando ajuste y test.
- `tests/posttraining/test_tabular_chain.py` (4): la etapa A publica una cadena para cada predictor que leen las políticas, Ridge y XGBoost solo planifican el padre congelado y su selección y B rechaza la cadena trivial. El recorrido usa la campaña A reducida con Ridge sustituido por una función fija de las presencias. El padre de fold-000 aplicado a fold-001 repite filas, objetivos y predicciones de la base en fold-001, sin cuantiles, y la selección publica ese padre como único candidato.
- `tests/training/test_carried_predictions.py`: el traslado tabular con `frozen_parent` predice también la validación de la ventana posterior y rechaza la ablación y la regeneración.
- `tests/training/test_campaign_throughput.py`, `test_campaign_plan.py`, `test_campaign_extensions.py` y `test_storage_budget.py`: recuentos de A (10.332 y 1.302), horas con filas nuevas, padre congelado y caché, el rechazo de recuentos sin filas nuevas, los brazos sin medir en `without_estimate` y una sola medida de A para las dos etapas, porque los casos de las redes de B están en A salvo por la huella de la matriz y A añade los de la variedad.

Diecisiete mutaciones dirigidas, aplicadas una a una sobre una copia del árbol, hacen fallar al menos una prueba cada una. Cubren la intersección, la madurez del padre, el inicio de las filas nuevas, el archivo de muestras y la huella de `staged_rows`, la desigualdad estricta, el desempate y el orden de la puntuación de `staged_chain`, el filtro y la lectura única de `ViewCohortSource`, la rejilla de `MatrixWindow`, la fase de ajuste de Titans-MAC y de la GRU candidata, la regla de las escalas M3 y, en la etapa, el filtro de filas nuevas, la ventana del padre y la comprobación de la puntuación declarada. En la primera pasada sobrevivieron dos: recorrer la validación en el orden escrito (la prueba aleatoria no distinguía los redondeos) y retirar la comparación entre la puntuación recalculada y la declarada. Se añadió una prueba para cada una y las dos fallan ahora.

### Comprobaciones CUDA

Ninguna de estas comprobaciones aplica pasos de optimizador. Se ejecutaron en `cuda:0` el 9 de octubre, con 5 y 3 pruebas superadas ([resumen](../../reports/engineering/cuda-checks-20261009/README.md)):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/models/test_predictive_adapters_cuda.py -q -rs
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/posttraining/test_quantile_adaptation_cuda.py -q -rs
```

La primera compara en `cuda:0` el padre y la copia con adaptadores nulos y contrasta con CPU con tolerancia relativa 1e-4 y absoluta 1e-5, sin TF32. En inferencia exige igualdad exacta. En entrenamiento exige pesos efectivos idénticos bit a bit y salidas con tolerancia relativa 1e-6 y absoluta 1e-7, porque `torch.matmul` resuelve la proyección de entrada no contigua de `MultiheadAttention` con `mm` cuando un peso requiere gradiente y con `bmm` cuando no, y en CUDA ambos núcleos redondean distinto (1,5·10⁻⁸ en el Transformer). La segunda calcula en `cuda:0` y en CPU la pinball de tres casos con padres de cuantiles (GRU con cabeza y fusión, Transformer con los tres puntos y continuación de LSTM) y compara niveles con las mismas tolerancias, pérdida y gradientes. Exige además el orden de los niveles, gradiente solo en los parámetros entrenables y el padre intacto. Antes de comparar gradientes comprueba en CPU que ningún nivel está a menos de 1e-4 del objetivo, donde la pinball cambia de pendiente. En CPU la distancia mínima de los tres casos es de 0,0093.

Los adaptadores de Titans-MAC tienen su propia comprobación explícita, ejecutada el 9 de octubre en la RTX 4070 Laptop GPU de 8 GB (controlador 595.91.07, PyTorch 2.14.0 con CUDA 13.0) con la reserva de la GPU de `memslot`, con 4 pruebas superadas en 34 s:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/posttraining/cuda_titans_adapters_check.py -q -rs
```

Usa float64, como los brazos de la campaña, sin TF32 y con el registrador de gradientes. En `cuda:0`, cada brazo de `transformer_direct` y `mac_online` con correcciones nulas emite exactamente las predicciones y el registro del padre congelado en el mismo dispositivo. Las predicciones del brazo con los tres puntos coinciden con las de CPU con tolerancia relativa 1e-8 y absoluta 1e-10, y el ajuste recorre los mismos pasos con gradiente solo en los adaptadores y valores iguales dentro de esas tolerancias.

Las formas de la variedad tienen su comprobación CUDA, que repite la identidad exacta y el contraste con CPU de cada forma en las cinco familias, y la de Titans-MAC recorre también los brazos de la variedad propuestos. Las dos pasaron el 10 de octubre en el mismo equipo sobre `develop` 3156ad1f, con 35 pruebas superadas y sin avisos de compactación de cuDNN:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/models/test_adapter_forms_cuda.py -q -rs
```

La cola y la etapa solo se han recorrido en CPU con los diagnósticos de `run_case`. Su recorrido en `cuda:0` con la reserva de la GPU no tiene todavía una comprobación propia.

## Pendiente

- Sustituir `labels_used_until` de `posttraining/staged_rows.py` por `training/label_maturity.py` de [#430](https://github.com/GonxKZ/mars-titan/pull/430), que lee el manifiesto de la vista en lugar del conjunto ya abierto, y adoptar en la carga de padres la política de precisión declarada de `perf/campaign-kernels`.
- Reconstruir en `posttraining/parents.py` el Transformer padre con el contrato de lote de `perf/campaign-kernels` (`transformer_batch_options`) cuando entre en `develop`, y rechazar un presupuesto cuyo lote supere el `max_batch` del padre. Hoy el lote de la matriz debe ser el de la campaña, que no pasa de 256.
- Decidir cómo se reduce el coste de lectura del orden exacto antes de ejecutar la etapa con la edición real, y medir cuántas filas nuevas y de validación caben en el presupuesto de cada ventana.
- Ejecutar la [comparación de los brazos postentrenados](../research/metrics.md#brazos-postentrenados) cuando existan sus predicciones. La declaración, la derivación por padre y la publicación del manifiesto de fuentes (`posttraining/stage_comparison.py`) están implementadas y comprobadas con recibos sintéticos. Falta declarar, antes de ver resultados, la comparación de los predictores de la cadena con el reentrenamiento base.
- Recorrer la cola y la etapa en `cuda:0` con la reserva de la GPU. Los lectores episódicos y la GRU candidata no tienen todavía una comprobación CUDA de sus adaptadores.
- Medir con la edición real el disco de los índices, las cachés de padres, los normalizadores y los checkpoints de cada ventana.
- `training.predictive_run` registra `fit_cutoff_utc` fijo en 2023. Es exacto para las dos particiones históricas, no para las ventanas. No se ha cambiado para no alterar su identidad estricta.
- Ejecutar la matriz tras verificar la edición y levantar el bloqueo, con coste medido antes. No hay mejoras predictivas medidas de ningún brazo.
