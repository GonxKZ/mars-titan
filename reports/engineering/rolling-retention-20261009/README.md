# Retención v2 de las predicciones por fila (9 de octubre de 2026)

Este informe acompaña a la retención v2 declarada en `configs/baselines/historical-masked-retention-v2.json` y a la [modificación del protocolo](../../../docs/research/protocol.md#modificación-del-9-de-octubre-de-2026-retención-v2-de-las-predicciones-por-fila) de la misma fecha. Recoge qué se ha implementado y comprobado, qué contiene el estado elegido que se conserva siempre y cuánto disco ocuparía la campaña A v2 recorrida ventana a ventana. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto. No se ha leído ningún objetivo ni modelo real. Las cifras usan GB decimales (10⁹ bytes).

## Resumen

- Una tabla por fila solo se libera si su regeneración desde el estado elegido sale idéntica bit a bit a su huella de contenido. Si no, se compacta sin pérdida y se conserva. En CPU, con los fixtures de las pruebas, la regeneración repite bit a bit las tablas de todas las familias de la campaña y la evaluación de la ablación. La comprobación en `cuda:0` está preparada y no se ha ejecutado.
- El estado elegido de cada familia guarda parámetros, optimizador, RNG, la selección y los normalizadores. No guarda memoria por flujo ni banco episódico, que se reconstruyen con el calentamiento de cada tramo, así que su tamaño no depende de las filas. En A v2 los estados elegidos suman 5,0 GB.
- Con la base, las políticas y la ablación, y si todas las regeneraciones salen idénticas, A v2 conserva 12,9 GB y su pico es de 42,2 GB en `fold-018`, con las páginas densas de XGBoost de `perf/campaign-tabular`. Con 48,0 GB libres y el margen de 8 GiB caben 39,4 GB, así que faltarían unos 2,8 GB en ese momento.
- Sin regeneraciones exactas todo queda compactado y la misma combinación conserva 112,6 GB. La comprobación en `cuda:0` decide, por tanto, si la campaña cabe.
- Los adaptadores de A v2 (1.653 ajustes) no caben: sus tablas compactadas conservan 118,7 GB. Liberarlas exige sus agregados por ventana y una regeneración propia, que no están implementados.

## Qué se ha implementado

| Pieza | Módulo | Comprobación |
| --- | --- | --- |
| Compactar y liberar tablas con lectura bit a bit y huellas | `data/prediction_files.py` | `tests/data/test_prediction_files.py` |
| Lectura de tablas compactadas o liberadas en todos los consumidores | `training/masked_campaign.py`, `modality_ablation_stage.py`, `titans_walk_forward.py`, `posttraining/campaign_stage.py`, `environments/window_tapes.py` | pruebas de cada consumidor |
| Agregados FP64 por ventana y fuentes de una ventana | `evaluation/window_aggregates.py`, `walk_forward_comparison.py --aggregates` | el informe final desde agregados es igual al que lee las filas |
| Regeneración por inferencia, sin ajustes y en FP32 estricto | `training/prediction_regeneration.py` y `regenerate=True` en cada traslado | `tests/training/test_prediction_regeneration.py` |
| Recorrido por ventanas, liberación verificada y guardia por ventana | `training/rolling_retention.py` | `tests/training/test_rolling_retention.py` |
| Boosters de recuperación de XGBoost e índices de la ablación tras el recibo | `training/campaign_storage.py`, `modality_ablation_stage.py` | `tests/training/test_campaign_storage.py` |
| Estimación del recorrido | `training/rolling_storage.py`, `run_masked_campaign.py storage --schedule` | `tests/training/test_rolling_storage.py` |

Los agregados guardan, para cada brazo y semilla de una ventana, los resultados por sesión que calcula la comparación (errores, aciertos y fallos direccionales por signo, Rank IC, pérdida de cuantiles, cobertura y anchura de intervalos, estratos de presencia, calibrador y ablación). Se escriben en un `npz` comprimido sin pérdida, con todos los decimales en FP64 y sin redondear, y la lectura comprueba su identidad con la configuración y las fuentes. La cartera larga y corta guarda aparte los libros por sesión de cada ventana (`<ventana>.long_short.npz`), con la identidad de la edición de precios. Su informe desde esos libros es idéntico al que lee las filas.

## Regeneración exacta

Un ajuste se regenera con el traslado de su familia sobre su propia ventana, con su intento como ancla: valida, calibra y evalúa de nuevo desde el estado elegido, con el mismo lote y el mismo orden, y escribe en un destino nuevo. La comparación con la tabla original usa la huella de contenido de `prediction_files`, que no depende de la disposición física, y además informa de si el archivo es idéntico byte a byte. El proceso fija `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False` y `float32_matmul_precision = "highest"`. Si el informe del ajuste registra otra política numérica, la regeneración se rechaza y la tabla se conserva. Cuando `feat/campaign-a-joint-design` se integre, esta fijación debe pasar a `campaign_numerics.apply`.

Resultados en CPU con los fixtures de las pruebas, que ajustan sin cambiar pesos (estado inicial de la semilla u optimizadores que solo registran gradientes):

| Familia | Fixture | Tramos | Resultado |
| --- | --- | --- | --- |
| Referencia neuronal (GRU) | campaña reducida de la ablación | validación, calibración y evaluación | idéntica bit a bit, también tras liberar las tablas |
| Ridge y XGBoost | escritor `_predict` con un modelo de recuento | los tres | idéntica bit a bit |
| Titans-MAC | ventana `mac_online` | los tres | idéntica bit a bit |
| MARS-TITAN M1 | lector sobre Titans-MAC | los tres | idéntica bit a bit |
| GRU episódica | ventana de la candidata | los tres | idéntica bit a bit |
| CM-v1 (B, B+C, B+M y B+C+M) | factorial reducido | los tres | idéntica bit a bit |
| Ablación de modalidades | evaluación enmascarada | evaluación | idéntica bit a bit |

Una tabla con un solo bit cambiado se detecta como distinta y la retención la compacta en lugar de liberarla. Trece mutantes dirigidos sobre la decisión de liberar, la comparación, la política numérica, las entradas de las políticas, la guardia, los agregados y la reanudación quedan detectados (`mutation.json`). Dos sobrevivieron en la primera pasada y motivaron dos pruebas nuevas. Los núcleos auxiliares de CM-v1 no tienen traslado, así que sus tablas no se liberan. La regeneración real de XGBoost carga el booster elegido con su propio predictor, que estas pruebas sustituyen.

Hay dos comprobaciones preparadas para la GPU, ninguna ejecutada. La primera ajusta en `cuda:0` Titans-MAC y, con el enlace nativo, el lector M1 de MARS-TITAN y la GRU episódica, en FP32 y FP32 estricto con optimizadores que solo registran gradientes, y exige que la regeneración en el mismo dispositivo repita bit a bit sus tres tramos. Su ensayo en CPU (`MARS_TITAN_REGENERATION_CHECK_DEVICE=cpu`) pasa con archivos idénticos byte a byte, lo que no acredita CUDA:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_EPISODIC_NATIVE=<enlace> \
  MARS_TITAN_REGENERATION_CHECK_REPORT=<informe.json> \
  uv run --no-sync pytest -q tests/training/cuda_regeneration_check.py
```

La segunda se hará con la campaña ya ejecutada, sobre un trabajo de cada familia:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run --no-sync python scripts/run_masked_campaign.py regenerate --campaign <configuración> \
  --views US=<vistas>/US --output <campaña> --job <trabajo de cada familia> \
  --destination <destino nuevo>
```

## Estado elegido que se conserva

`state-composition.json` recoge el estado elegido de cada familia en los fixtures. Contiene `model` (parámetros), `optimizer`, `rng` (10.048 bytes), la selección, el historial y, en la referencia neuronal, `statistics` con los normalizadores. MARS-TITAN y CM-v1 guardan un estado compuesto que nombra por huella el estado elegido de su padre. Lo que no es tensor ocupa entre 15 y 41 KB. En los fixtures los optimizadores solo registran gradientes, así que no hay momentos. Con AdamW en FP32 cada parámetro ocupa 12 bytes, la cifra de la declaración de almacenamiento, medida en un estado real de 50.881 parámetros.

Estados elegidos de A v2 con la declaración (un estado por ajuste tras liberar):

| Modelo | Ajustes | Estados (GB) |
| --- | ---: | ---: |
| Referencias neuronales | 508 | 0,89 |
| Ridge | 57 | 0,00 |
| XGBoost | 228 | 2,15 |
| GRU episódica | 57 | 0,37 |
| Titans-MAC | 432 | 0,97 |
| MARS-TITAN | 584 | 0,17 |
| Núcleos de CM-v1 | 152 | 0,38 |
| CM-v1 | 304 | 0,09 |

El tamaño de XGBoost es la cota del inventario para profundidad 6 y 2.000 rondas, sin medir, porque medirlo exigiría ajustar.

## Estimación de A v2

`estimate_a_v2.py` aplica `rolling_storage.rolling_estimate` al plan, al orden por ventanas y a las etapas de A v2 de `feat/campaign-a-joint-design` (`4f04aebb`), con sus recuentos por ventana, los bytes por fila medidos en el [presupuesto de disco](../campaign-storage-20261009/README.md) y sus medidas de agregados, adaptadores y cintas. `estimate-a-v2.json` guarda todas las cifras y cada ventana. La variante `tabular_dense_pages` usa la huella de XGBoost de `perf/campaign-tabular` (`496cc512`): páginas densas medidas de 1.719 columnas y la validación en RAM. `develop_xgboost_cache` usa la de `develop`, con páginas de 1.686 columnas y caché de validación en disco.

Referencia sin recorrido por ventanas (disposición actual):

| Etapa | Conservado sin liberar (GB) | Conservado con tabla común (GB) |
| --- | ---: | ---: |
| Base (2.322 ajustes) | 277,5 | 59,6 |
| Ablación (3.534 predicciones) | 183,3 | 77,5 |
| Adaptadores (1.653 ajustes) | 256,4 | 118,7 |

Recorrido por ventanas con liberación verificada:

| Etapas | Variante XGBoost | Conservado, todo regenerado (GB) | Pico, todo regenerado (GB) | Conservado, nada regenerado (GB) | Pico, nada regenerado (GB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | páginas densas | 7,7 | 34,5 | 57,1 | 79,9 |
| Base y políticas | páginas densas | 12,0 | 41,3 | 61,2 | 83,6 |
| Base, políticas y ablación | páginas densas | 12,9 | 42,2 | 112,6 | 131,0 |
| Base, políticas y ablación | caché de `develop` | 12,9 | 48,9 | 112,6 | 137,7 |
| Todo, adaptadores por bloques | cualquiera | 131,7 | 170,5 | 231,4 | 259,3 |
| Todo, adaptadores con copia ordenada | cualquiera | 131,7 | 317,4 | 231,4 | 406,2 |

Lo conservado con todo regenerado son los estados elegidos (5,0 GB), informes y resúmenes (0,9 GB), agregados (1,8 GB), registros de retención, las cintas de las políticas (4,0 GB) y las 418 evaluaciones que leen las políticas mientras tienen lector. El pico está siempre en `fold-018`, la ventana con más filas: lo conservado de las 18 anteriores (15,0 GB) más el ajuste XGBoost de US+CN con sus páginas (25,2 GB densas, 31,9 GB con la caché de `develop`). La regeneración de un trabajo al liberar añade sus tablas, su compactación y, si indexa, su índice, por debajo de ese pico.

A las 20:55 del 9 de octubre `/` tenía 47,98 GB libres. Con el margen de 8 GiB (8,59 GB) quedan 39,4 GB, menos que los 42,2 GB del pico de la base con políticas y ablación. Para lanzar esa combinación harían falta unos 50,8 GB libres en `fold-018`. La guardia de cada ventana lo comprueba con la declaración antes de empezarla, y la de cada trabajo antes de cada ajuste.

## Límites y trabajo pendiente

- La comprobación de regeneración en `cuda:0` no se ha ejecutado. Si una familia no repite sus bits en la GPU, sus tablas quedan compactadas y la estimación se acerca a la columna «nada regenerado».
- El recorrido necesita el filtro `window` de la base y de las etapas de `feat/campaign-a-joint-design`. Sin él `rolling` se niega a empezar.
- La comparación de las etapas de adaptadores lee filas. Necesita su registro de agregados por ventana, y los adaptadores su propia regeneración, antes de que se libere nada que lea. Hasta entonces sus tablas solo se compactan. Actualización del 10 de octubre: la fase de agregados ya guarda los de esa comparación antes de liberar la base, con sus bytes medidos ([informe](../campaign-publication-20261010/README.md)). La regeneración propia de los adaptadores sigue pendiente.
- La estimación cuenta las tablas comunes de filas hasta el final aunque se borren antes, y la medida declarada de la guardia es una cota superior de las tablas compactadas.
- Las cintas y estados de los ejecutores de PPO y KLPO no se pueden estimar todavía.
