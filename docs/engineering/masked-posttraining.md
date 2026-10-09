# Postentrenamiento con la edición histórica y matriz de adaptadores

Este documento describe cómo el postentrenamiento lee la edición `historical_masked_2000_v1` y cómo se declara la matriz finita de adaptadores de [#364](https://github.com/GonxKZ/mars-titan/issues/364), con los controles de [#128](https://github.com/GonxKZ/mars-titan/issues/128). Es una preparación técnica. No se ha ajustado ningún brazo ni se ha ejecutado ningún paso de optimizador. El bloqueo de aprendizaje sigue vigente y la reserva de 2024 permanece cerrada.

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

Un padre con la [cabeza de cuantiles](quantile-head.md) se reconstruye con `head: quantile_head_v1` y exige `output_head` con el contrato de la cabeza y la huella de `models/quantile_head.py`. Su predicción puntual es la mediana, que alimenta la corrección lineal y el control del padre congelado. La continuación completa y los adaptadores de la matriz optimizan un centro escalar, así que se rechazan con un padre de cuantiles en lugar de reinterpretar su pérdida pinball.

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

Cada semilla añade tres controles comunes: el padre congelado sin actualizaciones, la corrección lineal residual inicializada a cero y la continuación supervisada de todos los parámetros. Adaptadores y continuación usan `neural_mae`. La corrección lineal usa `mae` sobre su centro continuo. Todos comparten semillas 42, 43 y 44, cinco épocas, lote 256, AdamW con tasa 0,0001 y `weight_decay` 0,01, recorte de norma 1 y la selección de versión 2 sin paciencia y con mejora mínima 0,00001. Con la misma población, cada brazo y cada control aplican `5 × Σ⌈n_c / 256⌉` actualizaciones, donde n_c es el tamaño de cada sesión. La selección conserva el mejor estado de las cinco evaluaciones o el padre de la época cero.

La métrica de selección es la del contrato actual del postentrenamiento, el MAE por sesión de la mediana de la rejilla. Cada recibo conserva también los errores del centro continuo y del padre. En la época cero el centro de todos los adaptadores y de la corrección lineal coincide exactamente con la salida continua del padre, pero la mediana de 21 acciones no tiene por qué hacerlo. Si el contraste entre brazos debe usar el centro continuo, hará falta una nueva versión de la selección declarada antes de ejecutar.

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

### Arquitecturas pendientes

La matriz declara también destinos para Titans-MAC (`mac.query_projection`, `mac.attention.out_proj`, `fusion.0` y `head`), con `mac.memory` y `mac.persistent` congelados, y para la lectura episódica de MARS-TITAN (`query_projection` y `value_projection`). Una prueba comprueba que existen como capas lineales en `FinancialPredictor` y `EpisodicReadout` y que ningún destino cae dentro de un estado congelado. Los parámetros persistentes y los pesos rápidos de Titans no se convierten en adaptadores. Su postentrenamiento necesita el entrenador cronológico, porque `FinancialPredictor` sella sus parámetros y avanza memoria por sesión. La GRU candidata vive en el runtime C++ y no tiene destinos en Python.

### Implementación

`ResidualDelta` y `LowRankDelta` son parametrizaciones de `torch.nn.utils.parametrize`. El tensor original queda como parámetro congelado y la red no cambia su código. `adapted_copy` copia el padre, congela todos sus pesos, añade las parametrizaciones y no modifica el padre recibido.

Con correcciones nulas la salida coincide exactamente con la del padre cuando ambos recorren los mismos núcleos. En CPU, el modo de autograd cambia la ruta de LSTM y de la atención también para el propio padre, con diferencias de hasta 1,5e-8 en el fixture. Por eso la paridad exacta se comprueba en inferencia y en entrenamiento por separado, y la validación siempre se calcula en `inference_mode`.

## Protección del aprendizaje

`run_case`, `run_queue`, `run_completion` y las etapas tabular y de postentrenamiento de la compleción llaman a `learning_blocked()` antes de abrir fuentes o crear salidas. Si la protección local está vigente fallan con `RuntimeError` y el mensaje «Bloqueo de aprendizaje vigente». La evaluación congelada no ajusta parámetros y no cambia. Estas llamadas son provisionales hasta que exista `require_learning_allowed` en `training/learning_hold.py`.

Las pruebas del bloqueo declaran una protección temporal en `tmp_path` mediante `MARS_TITAN_TRAINING_HOLD`. Para que el resto de pruebas del paquete lleguen hasta el paso del optimizador como antes, `tests/posttraining/conftest.py` sustituye la protección que leen los puntos de entrada por una que los admite. Solo lo hace cuando la protección real está vigente, es decir, cuando `tests/conftest.py` ya ha registrado el gancho que omite cualquier paso de un optimizador de PyTorch.

## Comprobaciones

Todas se ejecutaron en CPU con `CUDA_VISIBLE_DEVICES=-1`, sin pasos de optimizador. Las pruebas del recorrido completo sustituyen AdamW por un optimizador que no hereda de `torch.optim.Optimizer`, registra parámetros y gradientes y exige que los pesos no cambien.

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

Comprobaciones CUDA pendientes, sin pasos de optimizador:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/models/test_predictive_adapters_cuda.py -q -rs
```

La prueba compara en `cuda:0` el padre y la copia con adaptadores nulos, exige igualdad exacta con los mismos núcleos y contrasta con CPU con tolerancia relativa 1e-4 y absoluta 1e-5, sin TF32.

## Pendiente

- Orquestar la matriz en la cola. Los casos y el plan existen, pero `run_queue` sigue ejecutando los ocho objetivos de #128 por padre.
- Conectar la edición con máscaras con el controlador temporal de las diez ventanas, que pertenece a otra tarea.
- `training.predictive_run` registra `fit_cutoff_utc` fijo en 2023. Es exacto para las dos particiones históricas, no para las ventanas. No se ha cambiado para no alterar su identidad estricta.
- Integrar los adaptadores de Titans-MAC y de la lectura episódica en el entrenador cronológico.
- Sustituir `refuse_while_blocked` por `require_learning_allowed` cuando se integre esa función.
- Postentrenamiento propio de los padres de cuantiles, con su pérdida pinball, si se decide estudiarlo.
- Ejecutar la matriz tras verificar la edición y levantar el bloqueo, con coste medido antes. No hay mejoras predictivas medidas de ningún brazo.
