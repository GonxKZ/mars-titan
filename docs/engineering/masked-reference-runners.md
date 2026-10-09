# Referencias neuronales con la edición histórica con máscaras

Este documento describe la adaptación de las referencias RNN, LSTM, GRU, DLinear y Transformer compacto a la edición histórica desde 2000 (`historical_masked_2000_v1`). El objetivo es que reciban las mismas filas, máscaras y objetivos que el núcleo Titans-MAC y que sus campañas conserven igualdad de actualizaciones. Es una preparación técnica. No se ha entrenado ningún modelo ni se ha evaluado la edición, que sigue pendiente de verificación. La reserva de 2024 permanece cerrada.

## Política de entradas

`run_reference_case` admite `input_policy`. Sin indicarla, o con `strict_inputs_v1`, se conserva la ruta estricta anterior. La política se transmite a `CorpusDataset(..., input_policy=...)`, también cuando `MARS_TITAN_INPUT_CACHE_MIB` activa la caché de tablas. Con la política histórica, la identidad de la ejecución añade `input_policy`, `mask_contract`, `mask_fusion` y la huella de `data/input_policy.py`. La continuación desde un padre exige la misma política, el mismo contrato de máscaras y la misma fusión.

La campaña de referencias y la búsqueda declaran la política en su configuración, que ya forma parte de su identidad:

| Componente | Versión | Campos nuevos |
| --- | --- | --- |
| `training.reference_campaign` | 3 | `input_policy`, `architecture`, `prediction_retention` |
| `training.reference_search` | 4 | `input_policy`, `stopping`, `prediction_retention` |

Las versiones anteriores se leen igual que antes. La campaña 2 sigue usando las sondas sin arquitectura y no puede leer la edición histórica. La versión 3 aplica una arquitectura científica declarada a todos sus casos, porque la fusión con presencia solo existe en `MultimodalReference`. Las vistas, los recibos de cada caso y la identidad científica de la campaña o búsqueda usan la política declarada.

## Fusión con presencia

`MultimodalReference` admite `mask_fusion="zero_after_projection_then_concat_presence"`, el mismo nombre que registra `FinancialPredictor`. El runner la activa siempre con la política histórica. La operación sigue el orden `prices`, `news`, `charts`, `fundamentals`, `macro` de `data.input_policy`:

1. El codificador de precios produce su representación sin máscara. El lector exige precios y gráficos presentes en cada fila.
2. Cada una de las otras cuatro modalidades se proyecta con su capa lineal y SiLU y después se multiplica por su bit de presencia. El sesgo de la proyección también queda anulado en un bloque ausente.
3. Los cinco bits se convierten al tipo de los pesos y se concatenan al final.
4. Solo la primera capa de fusión cambia de anchura, de `5D` a `5D + 5`.

La identidad de la referencia es distinta de la estricta. La inicialización no reproduce la de `FinancialPredictor`, que construye primero la capa estricta y después la sustituye. La equivalencia comprobada es la de la operación: con los mismos pesos copiados, la referencia Transformer y el control `transformer_direct` del adaptador financiero producen exactamente las mismas predicciones en float64.

Sin la opción se mantienen formas, consumo del generador, salidas y configuración anteriores. La fusión estricta rechaza bits de presencia y la fusión con presencia exige un tensor booleano de forma `[B, 5]` en el mismo dispositivo.

## Transformer compacto en el runner

`kind="transformer"` se admite con una arquitectura que incluye `transformer` con `heads` y `feedforward_multiplier` explícitos. El runner rechaza lotes mayores de 256 antes de leer datos y comprueba `B × T² × heads × layers ≤ 2²⁴` con el contexto de la edición antes de crear la salida. La identidad añade la huella de `models/baselines/transformer.py` solo en estos casos. Así, las identidades de las demás familias no cambian sus campos.

La búsqueda solo acepta la familia en la versión 4. El diseño fija cuatro cabezas y una FFN de anchura `2D`, las mismas opciones del codificador de `FinancialPredictor`, y conserva los doce niveles de anchura, profundidad, dropout, pérdida y tasa de aprendizaje de las otras familias.

## Presupuesto fijo y selección

La selección admite `stopping`. Sin el campo, o con `validation_plateau`, se mantiene la parada por meseta actual. Con `fixed_budget` se recorren todas las épocas declaradas y se conserva el mejor estado según `session_mae` y `min_delta`. La paciencia y la mejora mínima siguen siendo obligatorias y se declaran antes de ejecutar. Solo una validación completa avanza el contador. El estado registra además `plateau_epoch`, la época en la que habría parado la meseta, como diagnóstico sin efecto sobre el número de actualizaciones.

Con la misma población, el mismo lote y el mismo número de épocas, cada brazo aplica `épocas × ⌈N_train / B⌉` pasos de optimizador. Una continuación puede conservar el padre como época 0 si los ajustes empeoran. En la versión 4 de la búsqueda, `stopping` se aplica a la búsqueda y a las continuaciones, y `continuation_selection` no puede declarar otra parada.

`stop_reason` se publica también cuando la selección declara `stopping`. Con presupuesto fijo vale `budget_exhausted`, `stopped_early` es falso y el recibo incluye `plateau_epoch`.

## Retención de predicciones

`prediction_retention` tiene dos valores:

| Política | Filas completas | Ajuste |
| --- | --- | --- |
| `full_train_validation_v1` | `train`, `validation` | Filas completas |
| `heldout_full_train_sessions_v1` | `validation`, `calibration`, `evaluation` | `train-sessions.parquet` |

La primera es la política anterior y sigue siendo la predeterminada. La segunda evita guardar una fila por muestra de ajuste. El resumen tiene una fila por mercado y sesión, con recuento, errores absoluto y cuadrático del modelo y los mismos errores de la predicción nula. Sus métricas incluyen `sample_ids_sha256` y `predictions_float32_sha256`, calculados sobre el recorrido determinista de evaluación. Esas huellas no dependen de las fronteras entre lotes y permiten auditar el resumen volviendo a evaluar el estado seleccionado. Cada archivo registra su tamaño en bytes.

Calibración y evaluación se predicen solo cuando la vista temporal las contiene y siempre después de cargar el estado seleccionado. La búsqueda sigue eligiendo con las predicciones de validación. `CorpusDataset` no tiene una partición de 2024, por lo que esta ruta no puede abrir la reserva final.

El tamaño en disco de la política nueva no se ha medido sobre la edición real. Con la estimación de la auditoría, unos 15 millones de filas de ajuste podían ocupar cerca de 1 GB por ejecución. El resumen depende del número de sesiones y no del número de activos.

## Configuración preparada

[La búsqueda histórica estadounidense](../../configs/baselines/historical-masked-reference-search-us.json) usa la versión 4 con las cinco familias, los candidatos 0 y 10, las semillas 42, 43 y 44, 30 épocas, paciencia 5 y mejora mínima 0,00001. Son los mismos valores de la [búsqueda temporal estricta](strict-temporal-search.md), salvo el lote de 256 para todas las familias, el presupuesto fijo y la retención nueva. Por ventana son 10 casos de búsqueda, 10 finalistas nuevos y 30 continuaciones. Con diez ventanas suman 500 ejecuciones.

Es una propuesta técnica sin coste medido. Antes de lanzarla hay que medir tiempo, memoria y disco por época sobre la población admitida y revisar el presupuesto. El controlador temporal acepta la versión 4, lee la política del plan y exige que las vistas declaren la misma. Con el [protocolo anual v2](../research/walk-forward-2000.md) el plan debe aplicar además su regla de parada.

## Comprobaciones

Las pruebas nuevas se ejecutan en CPU con `CUDA_VISIBLE_DEVICES=-1`. Para recorrer el runner completo sustituyen `require_cuda`, las funciones de memoria CUDA y AdamW dentro de cada prueba. El sustituto de AdamW no hereda de `torch.optim.Optimizer` y no aplica actualizaciones. Registra los gradientes de cada llamada y exige que los pesos sigan iguales a los recibidos al construirlo. La guarda global de `tests/conftest.py` sigue omitiendo cualquier prueba que intente un paso real de un optimizador de PyTorch mientras el bloqueo esté vigente.

- `tests/models/test_masked_reference_fusion.py`: equivalencia exacta con el adaptador financiero, gradiente nulo y relleno irrelevante en bloques ausentes, proyección sin gradiente si su modalidad nunca se observa, paridad estricta para las cinco familias y rechazo de presencias mal formadas.
- `tests/training/test_selection_budget.py`: presupuesto fijo frente a meseta, `min_delta`, `minimum_epochs`, padre como época 0 y estado anterior sin campos nuevos.
- `tests/training/test_masked_reference_run.py`: lectura con máscaras, gradiente nulo en noticias y fundamentales ausentes en el fixture, mismas filas en validación, calibración y evaluación, resumen del ajuste, igualdad de pasos con presupuesto fijo, pausa y reanudación, continuación desde el padre y paridad de la ruta estricta.
- `tests/training/test_masked_reference_campaigns.py`: configuración del repositorio, búsqueda y campaña con la política histórica y campaña estricta sin campos nuevos.

Estos cuatro archivos suman 72 casos CPU. Veinte mutaciones dirigidas sobre la fusión, la selección, la identidad, la retención, la continuación y las configuraciones hacen fallar al menos una prueba cada una.

La paridad con la versión anterior del código se contrastó fuera de la suite, en dos procesos con cada árbol. Las 90 combinaciones de familia, profundidad, anchura y dropout conservan pesos, RNG, salidas, gradientes y configuración. Tres recorridos estrictos del runner (sonda DLinear, GRU con selección y LSTM con mínimo de épocas) conservan identidad salvo las huellas de código, claves del recibo, archivos de predicciones, métricas, selección y gradientes de cada llamada al optimizador.

Las 731 pruebas comunes de los módulos relacionados dan el mismo resultado en `develop` y en esta rama, salvo dos pruebas actualizadas. Una comprobaba que el diseño rechazaba el Transformer y otra sustituía `campaign_views` con la firma anterior. Los fallos compartidos por ambos árboles se deben a la GPU oculta, a `CUBLAS_WORKSPACE_CONFIG` o a vistas temporales que ya fallan en `develop`.

Las comprobaciones CUDA no se han ejecutado todavía. `tests/training/test_masked_reference_cuda.py` contrasta la fusión en `cuda:0` con CPU para las cinco familias y recorre el runner con la política histórica en `cuda:0`, con el mismo sustituto sin actualizaciones. Sin GPU sus siete casos se omiten. Con la GPU libre:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/training/test_masked_reference_cuda.py \
  tests/models/test_multimodal_reference.py tests/models/test_compact_transformer.py -q -rs
```

Las pruebas CUDA existentes de `test_reference_run.py`, `test_selection.py` y `test_corpus_reference_campaign.py` aplican pasos de AdamW y quedan bloqueadas hasta levantar la restricción de aprendizaje.

## Pendiente

- Decidir el presupuesto del protocolo anual v2. Con 50 trabajos por ventana, las 19 ventanas US y las 13 conjuntas superan el límite de 512 de `temporal_search`.
- `posttraining.parents` lee ya `mask_fusion` de la identidad del padre, como describe el [postentrenamiento con la edición histórica](masked-posttraining.md).
- La cola de referencias, el análisis de campañas y las fuentes de comparación esperan `train` y `validation` como predicciones completas. Deben aceptar la retención nueva antes de consumir estas ejecuciones.
- Ejecutar la campaña tras verificar la edición histórica y levantar el bloqueo, con medidas de coste previas. Las mejoras predictivas de cualquier familia siguen sin medir.
