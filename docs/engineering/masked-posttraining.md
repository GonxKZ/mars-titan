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

## Ejecución de la matriz

`posttraining/matrix_runs.py` reúne las piezas que comparten la cola y la etapa de la campaña. Ninguna decide qué padres o ventanas se recorren.

| Pieza | Función |
| --- | --- |
| `MatrixWindow` | Prepara una vez la lectura de ajuste y validación de una vista con la política de la matriz, el índice de cohortes de la [lectura por bloques](#lectura-por-bloques) o la copia ordenada, la reanuda si existe y abre sus dos lectores |
| `MatrixParent` | Carga el padre, abre su caché de predicciones, ajusta el normalizador solo con el tramo de ajuste y registra `plan.json` antes del primer ajuste. `run` llama a `run_case` y exige las actualizaciones del plan |
| `predict_heldout` | Reconstruye el estado seleccionado y escribe calibración y evaluación con las filas completas de la vista |
| `release_ordered` | Retira las copias Parquet de ajuste y validación que declara el manifiesto, después de comprobar su huella, y conserva el manifiesto |

### Presupuesto

`plan.json` guarda la huella de la matriz, la cabeza del padre, las actualizaciones por época de su ventana, los controles excluidos y una fila por caso con filas, actualizaciones, parámetros entrenables, bytes de pesos y momentos y estados invalidados. La primera fila es el padre congelado, con cero actualizaciones. `adapter_matrix.plan` exige el mismo número de actualizaciones en todos los brazos y controles del padre, y `MatrixParent.run` rechaza un ajuste completado que aplique otro número. Al reanudar, un plan distinto del registrado detiene la ejecución con «El plan de la matriz de este padre ha cambiado». Como cada ventana tiene otra población, el número cambia entre ventanas, pero no entre los casos de un mismo padre.

### Cola

`run_queue` decide el modo por el `kind` de la configuración. Un diseño de #128 sigue el recorrido anterior, con los ocho objetivos de cada padre y el resumen `paired_posttraining_queue` o `real_continuations_queue`. Una matriz (`posttraining_adapter_matrix`) pasa a `run_matrix_queue`, que escribe el resumen `adapter_matrix_queue`. Los dos modos no comparten salida ni identidad, y `read_design` sigue rechazando una matriz.

En el modo matriz, los padres son los ganadores y finalistas emparejados por semilla de la campaña de referencias de una vista (`matching_parents` con las semillas de la matriz), solo de las familias neuronales. La cola prepara una ventana con la vista de la supervisión y recorre cada padre y semilla con `MatrixParent`. La cabeza de cada padre se lee de su recibo para contar los casos previstos: un padre de cuantiles tiene cinco casos por semilla en RNN, LSTM, GRU y DLinear (cuatro brazos y la continuación) y nueve en el Transformer, y uno escalar añade la corrección lineal. El resumen confirma cada ejecución con la huella de su recibo y sus actualizaciones, junto con el plan de cada padre. Una matriz o un código cambiados, o un caso pendiente al final, detienen la cola. Los puntos de control de recuperación se escriben cada 300 s.

### Lectura por bloques

La copia ordenada de los tramos de ajuste y validación duplicaba en disco cada ventana. En la ventana US+CN más poblada (14,64 millones de filas de ajuste) eran unos 160 GB transitorios, según la medida de [#425](https://github.com/GonxKZ/mars-titan/pull/425). `environments/view_cohorts.py` evita esa copia sin cambiar los lotes:

- `prepare_cohort_index` recorre una vez cada tramo con el mismo lector que alimentaba la copia (`CorpusDataset.batches` con época y semilla cero, cuyo contenido no depende del tamaño de lote) y guarda solo el índice: sesiones y filas por sesión, mercados y límites, las comprobaciones de cada cohorte y la rejilla de acciones, ajustada con los objetivos en el mismo orden que la consulta de la copia (instante y activo). Su identidad incluye la huella de la vista, la política, la supervisión y el código. Un índice de otra vista, política o código se rechaza.
- `ViewCohortSource` entrega las cohortes en el orden de las visitas de cada época, la permutación del generador `[seed, epoch]`. Agrupa visitas consecutivas en bloques que caben en `max_block_bytes`, lee el tramo una vez por bloque y conserva solo las filas de las sesiones del bloque. Un vector nulo o idéntico al primero de su sesión (el macro de una sesión o las noticias ausentes) no se copia y se reconstruye con los mismos bits, también el signo de los ceros. Si un bloque supera el presupuesto mientras se lee, sus últimas sesiones pasan al siguiente, de modo que cada bloque es un prefijo completo de visitas.

Las pruebas comparan las dos lecturas sobre una vista conjunta US+CN con presencias distintas por mercado. Coinciden bit a bit el índice y la rejilla, cada cohorte con sus bits de presencia en las dos particiones y con varios presupuestos, también con bloques desalojados, los lotes y cursores del adaptador en dos épocas y en validación, la reanudación a mitad de cohorte y la normalización. Una etapa completa con cada lectura aplica los mismos gradientes en cada llamada al optimizador y escribe las mismas predicciones. Al probarlo apareció un fallo anterior: `PairedInputs` no admitía una fuente con dos mercados, que no tiene un límite único de tramo. Ahora cada fila se comprueba con los límites de su mercado.

Medida local en CPU sobre el corpus técnico conjunto, con 16 activos, gráficos distintos por fila y 3.984 filas de ajuste:

| Lectura | Disco tras preparar | Pico de disco | Pico RSS | Lecturas del tramo de ajuste en tres recorridos |
| --- | ---: | ---: | ---: | ---: |
| Copia ordenada | 8,6 MB | 33,6 MB | 955 MB | 3 |
| Bloques de 4 MiB | 37,7 kB | 0,25 MB | 792 MB | 9 |
| Bloques de 32 MiB | 37,7 kB | 0,25 MB | 794 MB | 3 |

El pico de disco de los bloques es la caché del padre, igual en las dos lecturas. Es una sola ejecución de cada caso sobre un corpus pequeño, sin la edición real ni CUDA.

El orden exacto tiene un coste de lectura. Cada bloque es una lectura completa del tramo de ajuste. En la ventana conjunta más poblada, la columna de presencia del tramo de ajuste tiene un 15,0 % de filas con noticias y un 32,6 % con fundamentales, lo que da unos 3.794 bytes por fila en un bloque. Con 3 GiB son 18 lecturas por época en esa ventana. Con el caudal del lector medido en `perf/campaign-pipeline` sobre vistas reales (de 17.000 a 25.000 filas por segundo en secuencia y unas 40.000 con hilos), cada lectura de esa ventana tarda entre 6 y 14 minutos. Es una estimación, no una medida de la etapa. Las alternativas (leer en el mismo recorrido todos los ajustes de una semilla y ventana, saltar las filas ajenas al bloque en el lector o declarar otro orden de visitas) están descritas en [#363](https://github.com/GonxKZ/mars-titan/issues/363) y ninguna está aplicada.

## Etapa por ventana de la campaña

`posttraining/campaign_stage.py` recorre la matriz sobre la [campaña con máscaras](../research/training-campaign-2000.md#orquestación-de-los-brazos-con-entrenador) ya confirmada. La etapa se declara en dos configuraciones, una por variante de presupuesto: [A](../../configs/posttraining/historical-masked-adapter-stage-a.json) y [B](../../configs/posttraining/historical-masked-adapter-stage-b.json). Cada una indica su campaña, la matriz de versión 2, los ámbitos, los brazos neuronales (RNN, LSTM, GRU, DLinear y Transformer compacto), la lectura de las cohortes y los límites de trabajos. `load_stage` exige que la matriz declare objetivos para `quantile_head_v1`, que comparta política y lote con la campaña, que los ámbitos sigan su orden, que los brazos pertenezcan a la comparación y que las semillas de cada brazo sean las de la matriz. Con la matriz de versión 3 admite también los [brazos cronológicos](#brazos-cronológicos).

| Elemento | Regla |
| --- | --- |
| Padre | Estado elegido del brazo base en la misma ventana y semilla, según los recibos confirmados de la campaña: el ganador de la búsqueda con la semilla 42 y el finalista con las demás. El informe debe conservar su huella |
| Ajuste | Solo el tramo de ajuste de la vista de esa ventana, con la lectura declarada, la caché y el normalizador de esa ventana y ese padre |
| Selección | Validación de la misma ventana, con la selección de versión 2 y el padre elegible en la época cero |
| Predicción | Calibración y evaluación de la vista con el estado seleccionado, el esquema común de la comparación, la mediana como predicción y los cinco cuantiles |
| Filas | Las de la vista. La huella de filas y objetivos de cada tramo debe ser la de la campaña base en esa ventana |
| Nombre del brazo | `<brazo base>__<punto>`, por ejemplo `gru__head_fusion` o `transformer_compact__full_continuation` |
| Padre congelado | Sin trabajos. Sus predicciones son las de la campaña base |

Cada trabajo confirma `jobs/<ámbito>/<ventana>/<brazo>/<fit|carry>-s<semilla>/receipt.json` con su identidad (etapa, caso, vista, trabajo base del padre, huella de su recibo y de su estado elegido y, en B, el ajuste del ancla), la ejecución, las actualizaciones, la selección y las predicciones con su huella, filas, huella de filas y objetivos y huellas por mercado. Antes de confirmarlo se comprueba que cada tramo cae en su segmento sin filas de 2024, que tiene las filas de la vista, que los cuantiles son finitos y no decrecientes y que la predicción es exactamente la mediana. Después se escribe el recibo walk-forward de cada mercado en `windows/<ámbito>/<ventana>/<brazo>/seed-<semilla>/<mercado>.json`, con el contrato de [#390](https://github.com/GonxKZ/mars-titan/pull/390), validado con `read_window_receipt` y con `labels_used_until` en el microsegundo anterior a la evaluación. Su `parent` es el trabajo de la etapa que ajustó el estado seleccionado y la huella de ese estado.

### Brazos cronológicos

Con la matriz de versión 3, un brazo de Titans-MAC, MARS-TITAN, CM-v1 o la GRU candidata entra en la etapa si la campaña base declara su sección. Uno que solo existe en la comparación queda en espera (`awaiting_sections` en `check`) y no genera trabajos. Sus casos salen de `chronological_matrix.cases` con la variante de Titans-MAC del brazo o con su banco, y los ejecutan `posttraining/chronological_windows.py` (Titans-MAC y lectores) y `posttraining/candidate_adapters.py` (GRU candidata):

- El ajuste parte del estado elegido del brazo base en la misma ventana y semilla, abre la vista de esa ventana y escribe validación, calibración y evaluación con el esquema común. No abre la lectura de cohortes de los brazos neuronales.
- El traslado de la variante B aplica sin ajuste el estado del ancla, con los mismos adaptadores, a la ventana trasladada. Cada intento escribe en una carpeta nueva (`attempt-0001`, `attempt-0002` y siguientes, hasta 16) y la etapa reutiliza el último si está completado.
- Un brazo que parte de otro predictor elegido, como MARS-TITAN de `titans_mac_online`, exige también los recibos confirmados de ese padre en la campaña base.
- Todos los ajustes de un mismo padre deben aplicar las mismas actualizaciones, y el recibo se rechaza si no.
- Tras confirmar un trabajo cronológico se borran sus índices de observaciones, que solo reconstruiría otra ejecución del mismo trabajo. El resumen registra los bytes liberados en `released_index_bytes`.

Las configuraciones A y B del repositorio todavía declaran la matriz de versión 2 y solo brazos neuronales. Declarar en ellas los brazos de Titans-MAC cambia recuentos, límites y caudal y queda pendiente.

### Variante B

En una ventana trasladada no se ajusta nada. El trabajo `carry` depende del ajuste del mismo caso en la ventana ancla y aplica su estado seleccionado, con el padre elegido en el ancla, a calibración y evaluación de la ventana trasladada. Es la misma regla con la que la campaña base traslada sus padres, y la etapa lo comprueba: el padre que la campaña base usa en esa ventana debe ser el del ancla. `carried_window` vuelve a comprobar que la información del ancla termina antes de la calibración de la ventana trasladada. El padre se carga con el índice o el manifiesto ordenado del ancla, que se conservan, y el recibo de la ejecución del ancla debe conservar su huella. Un traslado registra cero actualizaciones y publica su recibo walk-forward con el `parent` del ancla.

### Recuento

Por ventana y semilla hay 29 casos: cinco en cada familia recurrente y en DLinear y nueve en el Transformer. Con tres semillas son 87 por ventana.

| Variante | Ventanas reentrenadas | Ventanas trasladadas | Ajustes | Traslados |
| --- | ---: | ---: | ---: | ---: |
| A | 45 | 0 | 3.915 | 0 |
| B | 17 | 28 | 1.479 | 2.436 |

Los límites de cada configuración son iguales a su plan, como en la campaña base. `check` los comprueba sin leer datos.

### Ejecución, recuperación y disco

`run_stage` llama a `require_learning_allowed` antes de abrir fuentes y otra vez antes de cada trabajo pendiente. Lee el estado confirmado de la campaña base con su misma identidad (configuración, vistas y código) y exige que estén confirmados todos los trabajos base de los ámbitos y brazos de la etapa. La salida debe quedar fuera de las vistas, de la salida de la campaña y de `dataset/`, y guarda su identidad en `stage.json`: etapa, campaña, matriz, política, `training_data`, variante, edición de cada mercado, huellas de las vistas y huellas del código. Una salida de otra etapa, otra campaña u otro código se rechaza.

Los trabajos se recorren en el orden del plan, con una sola ventana, un padre y una vista abiertos. Un trabajo con recibo e identidad iguales no se repite, tras comprobar las huellas de sus artefactos. Uno pendiente se reanuda desde su punto de control con la frecuencia de la campaña. Una parada deja el resumen en `paused`, la protección en `blocked` y un error en `failed`. El resumen registra trabajos previstos y completados, las actualizaciones de cada trabajo, el plan de cada padre y las copias retiradas.

La declaración `cohort_reading` fija cómo leen los brazos neuronales:

| Lectura | Declaración | Disco por ventana |
| --- | --- | --- |
| Por bloques | `{"source": "view_blocks", "max_block_bytes": n}`, con n entre 256 MiB y 16 GiB | Índice de cohortes en `windows-data/<ámbito>/<ventana>/cohorts/` |
| Copia ordenada | `{"source": "ordered_corpus", "retention": "keep"}` o `"release_after_window_fits"` | Parquet de ajuste y validación. Con la retirada, se borran al confirmar todos los ajustes de la ventana y se conserva el manifiesto |

Las dos configuraciones del repositorio leen por bloques con 3 GiB. Cada ventana guarda además, por brazo base y semilla, la caché del padre, el normalizador y el plan. Un traslado carga el padre del ancla con el índice o el manifiesto de esa ventana, que se conservan. `training/storage_budget.py` solo cuenta la copia ordenada y su preparación cuando la etapa la declara.

```bash
uv run --no-sync python scripts/run_masked_campaign.py posttraining check \
  --stage configs/posttraining/historical-masked-adapter-stage-a.json
uv run --no-sync python scripts/run_masked_campaign.py posttraining run \
  --stage <configuración> --views US=<vistas>/US --views CN=<vistas>/CN \
  --views US+CN=<vistas>/US+CN --campaign-output <campaña> --output <etapa>
```

La etapa está registrada en `LATER_STAGES` de `training/campaign_plan.py` con la matriz de versión 2, la configuración de cada variante, su punto de entrada (`campaign_stage:run_stage`) y ninguna tarea pendiente, y `check` de la campaña la informa así. La orden única de la campaña la ejecuta con el subcomando `posttraining`, que delega en `campaign_stage.main`. La orden propia del módulo sigue disponible con los mismos argumentos. La [medición de caudal](../research/training-campaign-2000.md#medición-de-caudal) de la campaña mide también sus casos y estima sus horas.

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

- `tests/posttraining/test_quantile_adaptation.py` (17 pruebas): la pinball solo llega a los adaptadores y deja intacto el padre, la continuación recibe gradiente en todos los pesos, el orden de los cuantiles se conserva con pesos aleatorios, cada caso de cuantiles llega hasta el paso del optimizador con las actualizaciones previstas, se selecciona con la mediana y resume el ajuste con ella, un objetivo que no corresponde a la cabeza se rechaza antes de crear el optimizador y un padre escalar conserva su objetivo e identidad.
- `tests/posttraining/test_adapter_matrix.py` (37): además de lo anterior, la versión 2, su plan con padres de cuantiles y diez disposiciones de objetivos inválidas.
- `tests/posttraining/test_matrix_queue.py` (3): el modo matriz recorre cada caso por padre y semilla, se pausa y se reanuda sin repetir pasos, aplica las mismas actualizaciones en todos los brazos y controles, detecta un plan alterado, no repite casos confirmados, mantiene separado el modo de #128 y se detiene con la protección antes de leer.
- `tests/posttraining/test_adapter_campaign_stage.py` (31 con las de la lectura por bloques): recuentos de A y B con las configuraciones del repositorio, traslados desde el ancla, diez declaraciones inválidas y el recorrido completo sobre una campaña base reducida con padres de cuantiles reales y pesos iniciales (dos ventanas US, un brazo GRU y la semilla 42). En A comprueba diez ajustes con las mismas actualizaciones por ventana, predicciones idénticas a las del padre elegido, recibos walk-forward válidos y copias retiradas. Con la lectura por bloques no hay copia ordenada y las dos lecturas aplican los mismos gradientes y escriben las mismas predicciones. En B comprueba cinco ajustes en el ancla y cinco traslados sin optimizador. También pausa y reanudación, la protección antes de leer, antes de un ajuste y antes de un traslado, y el rechazo de filas de 2024, objetivos distintos, cuantiles desordenados, una predicción distinta de la mediana, un trabajo base sin confirmar y un traslado que no parte del padre del ancla.

Treinta y siete mutaciones dirigidas sobre el objetivo pinball, la evaluación de cuantiles, la evaluación reservada, la matriz de versión 2, la ejecución por padre, la cola y la etapa hacen fallar al menos una prueba cada una. En la primera pasada sobrevivieron cinco: retirar la protección por trabajo (`run_case` la comprueba también, pero un traslado no pasa por él), la comprobación de la mediana publicada, la de los objetivos de cuantiles al cargar la etapa, admitir `neural_mae` para padres de cuantiles y resumir el ajuste con los cinco niveles en lugar de la mediana. Se añadió una prueba para cada una y las cinco fallan ahora.

La paridad con `develop` (69bc59c1) se contrastó fuera de la suite con el mismo guion en los dos árboles y el optimizador registrador. Recorre los seis objetivos lineales y las dos continuaciones neuronales de la ruta estricta de #128, nueve casos de la matriz de versión 1 con máscaras sobre GRU, DLinear y Transformer, su calibración y evaluación, la evaluación congelada de cada padre y los diseños existentes de la cola. Coinciden las 639 llamadas al optimizador y la huella de todos sus gradientes, los Parquet de validación, calibración y evaluación, las métricas, los casos y los recibos. Solo cambian las huellas de código, los checkpoints que las contienen y la huella del manifiesto ordenado, que incluye tiempos de ejecución y también cambia entre dos ejecuciones de la misma rama.

Después de rebasar sobre `develop` (6119a7ea), 39 archivos de pruebas de postentrenamiento, campaña, referencias con cuantiles, recibos walk-forward y comparación dan 585 pruebas superadas, 55 omitidas y ninguna fallida, ejecutados uno a uno. Las omisiones son 42 por el bloqueo (pasos reales de AdamW y campañas de referencias) y 13 por la ausencia de CUDA. `ruff check`, `ruff format --check` y `scripts/check_repository.py` no dan errores.

### Familias cronológicas, lectura por bloques y datos reales

- `tests/posttraining/test_adapter_matrix.py` (54): además de lo anterior, la versión 3, sus casos cronológicos con receta y semillas de componente fijas y las declaraciones abiertas o que tocan la memoria.
- `tests/posttraining/test_titans_adapters.py` (14): con correcciones nulas, cada brazo de las cuatro variantes emite los mismos bits que el padre congelado, los recuentos coinciden con los tensores declarados, el gradiente solo llega a los adaptadores y la memoria queda congelada, los brazos de un padre comparten actualizaciones, etiquetas y predicciones, el gradiente de cada corrección es la regla de la cadena aplicada al de la continuación completa, la reanudación reproduce el recorrido continuo y sin postentrenamiento los papeles y la identidad no cambian.
- `tests/posttraining/test_readout_adapters.py` (12): brazos de MARS-TITAN y CM-v1 con los mismos eventos, actualizaciones y predicciones emitidas, gradiente solo en las correcciones declaradas, regla de la cadena de la lectura, gradiente del núcleo independiente del tamaño de bloque de flujos, núcleo sin banco, penalización C sobre el operador adaptado, reanudación y rechazos.
- `tests/posttraining/test_candidate_adapters.py` (5): la cabeza en Python reproduce bit a bit la cabeza nativa, el brazo emite las filas del padre en los dos mercados, solo se mueven las correcciones de la cabeza, el traslado coincide con el del padre y se rechazan casos de otra matriz o semilla.
- `tests/posttraining/test_readout_windows.py` (3): sobre una campaña B reducida con `titans_mac_online` y `mars_titan_m1`, los cuatro casos de M1 ajustan y trasladan con las filas y los bits del brazo base y las mismas actualizaciones.
- `tests/posttraining/test_chronological_stage.py` (5): la etapa planifica, ajusta y traslada `titans_mac_online` con la continuación y la cabeza sobre la campaña B reducida de Titans-MAC (14 ajustes y 24 traslados), reproduce las filas de la base, libera los índices y no repite trabajos al reanudar.
- `tests/environments/test_view_cohorts.py` (17): índice, rejilla, cohortes, lotes, cursores, reanudación y normalización iguales bit a bit a los de la copia ordenada, con desalojo, y rechazos de índices ajenos.
- `tests/posttraining/test_real_data_only.py` (17): las importaciones del programa de la campaña, la ejecución de `posttraining check` con los módulos de #128 bloqueados, siete declaraciones ajenas a los datos reales, la campaña sin máscaras, la edición de cada mercado en la identidad, entradas reales sin episodios y el orden de las cohortes reales igual al anterior.

Las pruebas de `tests/posttraining/test_adapter_campaign_stage.py` y `test_chronological_stage.py` recorren la etapa con los módulos de #128 sustituidos por centinelas.

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

La cola y la etapa solo se han recorrido en CPU con los diagnósticos de `run_case`. Su recorrido en `cuda:0` con la reserva de la GPU no tiene todavía una comprobación propia.

## Pendiente

- Sustituir la etapa por ventana por el walk-forward por etapas de la campaña A: padre elegido en la ventana anterior, ajuste solo con las filas de `train_k` que el padre no usó, selección con `val_k`, predictor de la cadena por ventana y sin variante B. Las secciones sobre la etapa describen todavía el diseño anterior.
- Adoptar `training/label_maturity.py` de [#430](https://github.com/GonxKZ/mars-titan/pull/430) en los recibos walk-forward de la etapa y la política de precisión declarada de `perf/campaign-kernels` en la carga de padres, cuando entren en `develop`.
- Decidir cómo se reduce el coste de lectura del orden exacto antes de ejecutar la etapa con la edición real.
- Declarar antes de ver resultados una comparación con los brazos postentrenados y publicar su manifiesto de fuentes para `evaluation.walk_forward_comparison`. La etapa escribe sus predicciones con el esquema común y los recibos walk-forward, pero no publica ese manifiesto.
- Recorrer la cola y la etapa en `cuda:0` con la reserva de la GPU. Los lectores episódicos y la GRU candidata no tienen todavía una comprobación CUDA de sus adaptadores.
- Medir con la edición real el disco de los índices, las cachés de padres, los normalizadores y los checkpoints de cada ventana.
- `training.predictive_run` registra `fit_cutoff_utc` fijo en 2023. Es exacto para las dos particiones históricas, no para las ventanas. No se ha cambiado para no alterar su identidad estricta.
- Ejecutar la matriz tras verificar la edición y levantar el bloqueo, con coste medido antes. No hay mejoras predictivas medidas de ningún brazo.
