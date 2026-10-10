# Publicación final de la campaña y agregados de la comparación postentrenada (10 de octubre de 2026)

Este informe acompaña a [#28](https://github.com/GonxKZ/mars-titan/issues/28). Recoge qué se ha implementado para que la matriz de comparaciones y la comparación postentrenada lean la comparación de su propia campaña, cómo se ha comprobado que el paso final calculado desde los agregados por ventana coincide con el que lee las filas y cuánto disco ocupan esos agregados en A v2. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto. No se ha leído ningún objetivo, vista real ni modelo. Las cifras usan GB decimales (10⁹ bytes).

## Qué se ha implementado

| Pieza | Módulo | Comprobación |
| --- | --- | --- |
| Escritor del manifiesto de fuentes de la matriz, con rutas relativas y validación antes de publicar | `evaluation/session_table_contrasts.py` (`write_sources`) | `tests/evaluation/test_comparison_matrix.py` |
| Matriz de A v2 sobre la comparación conjunta, con los brazos de integración pendientes de #437 | `configs/evaluation/comparison-matrix-a-v2.json` | `tests/training/test_campaign_publication.py` |
| Declaración del paso final de A y de A v2 y su validación cruzada | `training/campaign_publication.py`, `configs/evaluation/historical-masked-publication-a*.json` | `tests/training/test_campaign_publication.py` |
| Agregados por ventana de la comparación de cada padre | `posttraining/stage_comparison.py` (`write_window_aggregates`, `--aggregates`) | `tests/posttraining/test_stage_comparison.py` |
| Fase de agregados del recorrido con la comparación postentrenada | `training/rolling_retention.py` (`--publication`) | `tests/posttraining/test_publication_from_aggregates.py` |
| Paso final de la campaña | `run_masked_campaign.py publication check|run` | las dos pruebas anteriores y `tests/training/test_rolling_retention.py` |
| Disco de esos agregados en la proyección por ventanas | `training/rolling_storage.py` (`comparison_series`) | `tests/training/test_rolling_storage.py` |

La publicación de cada campaña se declara antes de ver resultados y nombra la campaña, su matriz y la comparación de su etapa de adaptadores. La carga exige que la matriz lea la comparación de la campaña con la misma huella y que la comparación postentrenada derive de una etapa de esa misma campaña. La matriz de A lee `historical-masked-2000-comparison.json` y la de A v2 `historical-masked-2000-joint-comparison.json`. Antes de este cambio solo existía la matriz de A. La campaña que se ejecutará, A v2, no tenía matriz sobre su comparación conjunta, y el manifiesto de fuentes de la matriz no tenía escritor.

## Equivalencia desde los agregados

Con la retención v2, las filas de calibración y evaluación del brazo base reentrenado se liberan al terminar cada ventana, y la comparación de cada padre las lee. La fase de agregados guarda ahora, para cada padre con trabajos de la etapa en la ventana, las fuentes de esa ventana y los agregados por sesión que calcula `_score_window` con la configuración del padre limitada a la ventana. La configuración limitada conserva la huella, así que la lectura final comprueba que cada archivo procede de las mismas predicciones, la misma configuración y el mismo código.

Todo lo que el informe calcula después de puntuar las ventanas (unión de sesiones, resumen por semilla, contrastes, bootstrap por bloques, corrección múltiple y fiabilidad) parte de esas puntuaciones por sesión. No hay ninguna parte que no se descomponga por ventana, y las pruebas exigen igualdad exacta, sin tolerancia:

- `test_the_comparison_from_window_aggregates_equals_the_one_that_reads_the_rows`: etapa de tres ventanas, de las que la comparación cubre dos. El informe desde agregados, con `_score_window` sustituido por una función que falla, es idéntico al que lee las filas, intervalos del bootstrap con la misma semilla incluidos, y la tabla por sesión también.
- `test_the_publication_from_aggregates_equals_the_reports_that_read_the_rows`: campaña A reducida, etapa de adaptadores ejecutada con un optimizador que solo registra gradientes y nunca cambia pesos, y recorrido de la retención v2 con la publicación. Tras liberar, la comparación postentrenada desde filas da `PredictionsReleased`. La publicación por la orden de la campaña reproduce exactamente la comparación walk-forward, la comparación del padre y la matriz calculadas antes de liberar. Solo cambia la huella del manifiesto de fuentes de la etapa, porque guarda rutas relativas a la copia recorrida. Sin rutas, el manifiesto es el mismo archivo a archivo.
- `test_the_campaign_publication_reads_the_aggregates_with_portfolio_and_ablation`: la misma publicación sin etapa de adaptadores, con la cartera larga y corta y la ablación de modalidades desde agregados.

## Orden y recuperación

Cada agregado se escribe de forma atómica, se relee igual y solo entonces se registra la fase en `retention/ledger.json`. La liberación empieza después. `test_a_cut_between_the_aggregates_and_the_release_loses_and_repeats_nothing` corta dos veces:

1. con los agregados escritos y sin registrar, las filas de la base siguen presentes y la reanudación repite los agregados en los mismos archivos;
2. con los agregados registrados y sin liberar, las filas siguen presentes y la reanudación libera sin reescribirlos (misma huella que el registro).

Al terminar hay un solo archivo por padre y ventana, ningún temporal en los agregados ni en las fuentes de la etapa, y la publicación coincide con la que leía las filas. La publicación escribe en una carpeta provisional marcada que solo toma el nombre del destino al terminar. Un fallo no deja destino y la ejecución siguiente descarta la carpeta marcada. Una carpeta sin la marca no se toca.

Quince mutantes dirigidos sobre estas decisiones quedan detectados (`mutation.json`).

## Bytes medidos

`measure_stage_aggregates.py` escribe con `window_aggregates.write`, la misma función que usa la fase de agregados, los agregados de los 20 padres de la [comparación de A v2](../../../configs/posttraining/historical-masked-adapter-comparison-a-v2.json) en `fold-018` de US+CN, la ventana con más filas y sesiones. Las predicciones son sintéticas con la forma real: 254.563 filas de calibración y 1.027.173 de evaluación en US y 47.414 y 194.649 en China, según [los recuentos de A v2](../../data/campaign-a-v2-window-counts-20261009.json), con las decisiones del calendario de cada mercado, hasta 4.126 activos por sesión, los brazos y semillas de cada padre y la cabeza de cuantiles. No sale ninguna de un modelo y no se lee ninguna vista real. Los estratos de presencia necesitan las muestras de la vista. Con `--strata` cada sesión reparte sus filas entre los cuatro estratos, el caso con más series por estrato. Con la edición real puede haber sesiones sin alguno, así que esa medida es una cota superior.

| Medida | Series | Agregados (bytes) | Fuentes de la ventana (bytes) | Bytes por serie | Tiempo | Memoria máxima |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Con los cuatro estratos ([medida](stage-aggregates-fold-018-strata.json)) | 546 | 644.155.610 | 383.621 | 1.180.421 a 1.180.646 | 3.288 s | 2,10 GB |
| Sin estratos ([medida](stage-aggregates-fold-018-without-strata.json)) | 546 | 134.960.601 | 383.621 | 247.865 a 247.914 | 2.666 s | 1,95 GB |

Los bytes por serie apenas cambian entre padres, porque cada serie guarda las mismas puntuaciones por sesión. Lo que más ocupa son las matrices por sesión de cada brazo y semilla (unas 490 sesiones de los dos mercados), en bruto, calibradas y por estrato. Los tiempos son de un proceso con dos hilos y la CPU compartida con otras cargas (carga media de 15 a 21).

`project_a_v2.py` cuenta las series de cada ventana con `rolling_storage.comparison_series` sobre el plan de la etapa de A v2 y comprueba que coinciden con las de la declaración: 546 en cada una de las 18 ventanas comparadas, 9.828 en total ([proyección](projection-a-v2.json)). Con los bytes por serie de `fold-018`:

| Agregados de la comparación postentrenada de A v2 | Por ventana | Todo el recorrido |
| --- | ---: | ---: |
| Con los cuatro estratos (cota usada por la guardia) | 0,64 GB | 11,60 GB |
| Sin estratos | 0,14 GB | 2,44 GB |

Es una cota superior también por otra razón: antes de `fold-006` China no entra en las métricas y cada ventana tiene la mitad de sesiones. No se ha medido una de esas ventanas.

## Suma a la proyección de disco

`rolling_storage.adapter_window` suma ahora a lo que conserva cada ventana con adaptadores sus series por `comparison_aggregate_bytes_per_series`, que [`extras.json`](../campaign-storage-20261009/extras.json) declara con la medida con estratos (1.180.647 bytes, redondeada hacia arriba). La guardia de cada ventana del recorrido y `run_masked_campaign.py storage --schedule` la usan desde este cambio. En la [estimación de A v2 del 9 de octubre](../rolling-retention-20261009/README.md#estimación-de-a-v2), las filas con todas las etapas (131,7 GB conservados si todo se regenera y 231,4 GB si nada se regenera, con adaptadores por bloques) no contaban estos agregados. Con ellos, lo conservado sube como mucho 11,6 GB en los dos escenarios, a unos 143,3 y 243,0 GB, y el pico de cada ventana con adaptadores sube como mucho lo acumulado hasta ella. Los adaptadores siguen dominando con sus tablas compactadas (118,7 GB). Esa estimación no se ha regenerado, porque dependía de las copias de dos ramas de aquel día.

## Límites y trabajo pendiente

- Los estratos de la comparación postentrenada se heredan de la comparación de la campaña y explican unos 9,2 GB de los 11,6 GB. Quitarlos de la declaración postentrenada antes de ver resultados dejaría unos 2,44 GB. Es una decisión de diseño que no se ha tomado aquí.
- Los agregados de la comparación de la campaña base siguen proyectados con 0,59 bytes por fila de la medida del 9 de octubre. Con el formato real y los cuatro estratos, la medida de hoy da unos 0,77 bytes por fila de calibración y evaluación en `fold-018`, sin la ablación, así que esa partida también puede quedarse corta. No se ha medido.
- Las tablas de los adaptadores siguen compactándose porque no tienen regeneración propia.
- El banco de integridad todavía no forma parte del paso final. Lo hará en [#442](https://github.com/GonxKZ/mars-titan/issues/442), antes de liberar cada ventana, porque `row_identity` y `score_recheck` necesitan las filas.
- La campaña no se ha ejecutado. No hay ninguna cifra de resultados.

## Reproducir

```bash
export PYTHONPATH=src
R=reports/engineering/campaign-publication-20261010
uv run --no-sync python $R/measure_stage_aggregates.py --root <carpeta nueva> \
  --window fold-018 --strata --output $R/stage-aggregates-fold-018-strata.json
uv run --no-sync python $R/measure_stage_aggregates.py --root <carpeta nueva> \
  --window fold-018 --output $R/stage-aggregates-fold-018-without-strata.json
uv run --no-sync python $R/project_a_v2.py --strata $R/stage-aggregates-fold-018-strata.json \
  --plain $R/stage-aggregates-fold-018-without-strata.json --output $R/projection-a-v2.json
```

Cada medida necesita unos 2,1 GB de memoria y 1 GB de disco temporal, que el script borra al terminar.
