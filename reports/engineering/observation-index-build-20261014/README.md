# Construcción de los índices de observaciones (14 de octubre de 2026)

Este informe acompaña a [#622](https://github.com/GonxKZ/mars-titan/issues/622). Recoge qué cambia en la construcción del índice de observaciones de cada tramo, la comparación de sus archivos de eventos con `develop` en vistas reales de A v2 y el tiempo y la memoria medidos. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto, y no se ha abierto el test final.

## Resumen

En las tres vistas medidas, los archivos de eventos de los cuatro tramos tienen la misma huella SHA-256 con `develop` (06e69db6) y con la rama (a6cbe69d) en todas las rondas. La construcción de los cuatro índices pasa de 51,9 y 52,1 s a 17,2 y 15,4 s en `CN/fold-012` (810 activos), de 331,5 y 325,7 s a 97,6 y 95,6 s en `US+CN/fold-000` (5.008 activos) y de 632,3 s a 238,8 s en `US+CN/fold-018`, la vista mayor. La relación entre la ronda más lenta de la rama y la más rápida de `develop` es de 3,02, 3,34 y 2,65. El pico de memoria del proceso también baja: de 1.446–1.503 MiB a 1.248–1.280 MiB, de 2.442 MiB a 1.162–1.167 MiB y de 3.798 MiB a 2.256 MiB.

Con el cambio, la validación de las etiquetas de cada activo (`CorpusDataset._labels`) pasa a ocupar un 54,6 % del hilo principal. Se trata en [#625](https://github.com/GonxKZ/mars-titan/issues/625).

## Qué cambia

Con #618 cada vista construye una vez el índice de cada tramo, pero los trabajos de la vista esperan a que termine, así que su duración sigue en el camino crítico. Un perfil de py-spy de los cuatro índices de `CN/fold-012` con `develop` (86 s, 810 activos) repartía el hilo principal así: `_records` un 57 % (`read_row_group` 15 %, `_labels` 12,5 %, `_historical_times` 9 %, `_table` 6 % y la apertura de cada archivo 5,5 %), la escritura del Parquet sin ordenar un 22 % y la ordenación con DuckDB un 14 %.

`_records` leía `prediction_at` grupo a grupo y emitía una tabla por grupo, también vacía. En `US+CN/fold-018` son 148.358 grupos de 126 filas de media en 5.008 activos, y el Parquet sin ordenar acababa con un grupo diminuto por cada uno. Ahora `_asset_samples` lee la columna de todo el archivo en una llamada y obtiene el grupo físico y la fila de cada muestra con los desplazamientos de los metadatos. Los grupos vacíos no aportan filas, así que el grupo sale igual que en la lectura grupo a grupo. El orden estricto de todas las filas del archivo equivale al de cada grupo más el del primer valor de cada grupo frente al último del anterior, y el corte 2000–2023 se comprueba sobre las mismas fechas.

`_records` reúne las filas de varios activos y entrega tablas de al menos `_RAW_ROWS` (262.144) filas, salvo la última, sin tablas vacías. Las filas salen en el mismo orden que antes: por activo, sus muestras y después sus maduraciones, y al final el cierre. Solo cambia cómo se reparten en grupos del Parquet sin ordenar, que DuckDB ordena después por una clave única (`event_at`, `kind`, `asset_id`, `sample_at`). Los límites (presupuesto de muestras por activo y número máximo de registros) se comprueban por activo, antes de leer sus etiquetas y antes de pasar al siguiente. Si un archivo tiene a la vez fechas fuera de orden y fuera del corte, ahora se informa primero del corte aunque el desorden aparezca en un grupo anterior, porque el corte se comprueba sobre todo el archivo. Los dos casos detienen la construcción.

La receta del índice no cambia. Su identidad incluye la huella del código, así que un índice construido con `develop` no se reutiliza con este cambio y se construye otro al lado.

## Archivos de eventos

[`summary.json`](summary.json) guarda, para cada ronda, el tiempo, las filas, los bytes y la huella SHA-256 del archivo de eventos de cada tramo. En cada vista, las cuatro huellas coinciden en todas las rondas de los dos árboles. En `CN/fold-012` y `US+CN/fold-018` coinciden también con los índices construidos antes con `develop` para las medidas de #618.

| Vista | Tramo | Registros | SHA-256 (inicio) |
| --- | --- | ---: | --- |
| `CN/fold-012` | ajuste | 4.908.804 | `f5f9abbf72a3` |
| | validación | 393.330 | `f47d8ffd2833` |
| | calibración | 291.789 | `8c7cb22eaef9` |
| | evaluación | 586.406 | `d4020a114a4a` |
| `US+CN/fold-000` | ajuste | 2.723.441 | `01ee6889a321` |
| | validación | 774.613 | `eb00f40b876d` |
| | calibración | 595.893 | `308bd3c39051` |
| | evaluación | 1.235.099 | `908c05cd2b7c` |
| `US+CN/fold-018` | ajuste | 32.723.183 | `c51014b70c99` |
| | validación | 2.494.136 | `848316c021fa` |
| | calibración | 1.866.623 | `d36986734083` |
| | evaluación | 3.721.361 | `8265490ad583` |

## Tiempo y memoria

Medidas del 14 de octubre de 2026 en el portátil del proyecto (AMD Ryzen 9 8945HS con 16 hilos, 32 GB y perfil de energía `power-saver`), cada construcción en un proceso nuevo dentro de su propio trabajo `memslot heavy` con `OMP_NUM_THREADS=2` y sin GPU. Las rondas siguen el orden A, B, B, A para repartir la deriva térmica y de caché. En `US+CN/fold-018` solo se midió una pareja A, B, porque cada ronda de `develop` dura más de diez minutos. El equipo tenía otras cargas ajenas al proyecto, y la carga media antes de cada ronda queda en `summary.json` (entre 1,6 y 3,0).

| Vista | Ronda | Árbol | Total (s) | Ajuste | Validación | Calibración | Evaluación | Pico RSS (MiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `CN/fold-012` | 1 | `develop` | 51,9 | 24,9 | 9,1 | 8,8 | 9,2 | 1.503 |
| | 2 | rama | 17,2 | 10,2 | 2,2 | 2,4 | 2,4 | 1.248 |
| | 3 | rama | 15,4 | 8,3 | 2,3 | 2,3 | 2,5 | 1.280 |
| | 4 | `develop` | 52,1 | 26,0 | 8,7 | 8,4 | 9,0 | 1.446 |
| `US+CN/fold-000` | 1 | `develop` | 331,5 | 128,7 | 66,4 | 66,3 | 70,1 | 2.442 |
| | 2 | rama | 97,6 | 49,0 | 17,2 | 15,5 | 15,9 | 1.162 |
| | 3 | rama | 95,6 | 50,0 | 15,4 | 14,8 | 15,4 | 1.167 |
| | 4 | `develop` | 325,7 | 129,8 | 68,0 | 64,3 | 63,6 | 2.442 |
| `US+CN/fold-018` | 1 | `develop` | 632,3 | 213,0 | 166,1 | 119,0 | 134,2 | 3.798 |
| | 2 | rama | 238,8 | 84,8 | 49,8 | 51,7 | 52,4 | 2.256 |

Un perfil de py-spy (50 muestras por segundo, todos los hilos) de la rama construyendo los tramos de ajuste y validación de `US+CN/fold-018` reparte el hilo principal así ([`profile-018.json`](profile-018.json), calculado con [`profile_shares.py`](profile_shares.py)): la escritura del Parquet sin ordenar con sus registros un 84,1 %, de la que `_records` es un 81,8 %, y la ordenación con DuckDB (`_ordered_parquet`) un 10,3 %. Dentro de `_records`, `_labels` ocupa un 54,6 % (un 16,2 % en su propio código, `np.unique` un 9,8 %, la asignación de la vista temporal un 9,3 %, la lectura de la tabla un 8,6 % y la comprobación de la cohorte un 4,5 %), la apertura de los archivos Parquet un 16,0 % y `_asset_samples` un 8,2 %. Con py-spy el proceso tarda más que sin él (135,6 s el ajuste y 89,3 s la validación), así que el perfil sirve para repartir el tiempo y no como medida.

## Pruebas y mutaciones

| Prueba | Qué comprueba |
| --- | --- |
| `tests/memory/test_observation_index_records.py::test_records_keep_the_rows_and_order_of_the_grouped_reading` | Los registros coinciden con la lectura grupo a grupo en archivos compactos, con un grupo vacío final y con grupos vacíos intermedios, con `_RAW_ROWS` de 1, 7 y 262.144, y ninguna tabla ni fragmento queda vacío |
| `test_assets_without_matured_labels_leave_no_empty_piece` | Un activo sin maduraciones en el tramo no deja una tabla vacía entre los registros |
| `test_inner_empty_groups_shift_the_physical_group_of_later_rows` | Los grupos vacíos intermedios desplazan el grupo físico de las filas posteriores y no su fila |
| `test_records_reject_samples_out_of_order` | Una fecha repetida dentro de un grupo, repetida entre grupos o anterior entre grupos se rechaza con el mismo error que antes |
| `test_records_reject_samples_outside_the_historical_cut` | Una muestra anterior a 2000 se rechaza |
| `test_records_stop_at_the_record_limit` | El límite de registros detiene la lectura en las muestras, sin leer sus etiquetas, o en las maduraciones, sin pasar al siguiente activo |

Las medidas se hicieron con la rama sobre `develop` 06e69db6 (a6cbe69d). Después se rebasó sobre dd0a5d68 (#619), que solo cambia el lector de bloques, y en el árbol rebasado pasan `tests/memory` completo (703 pruebas) y los archivos de entrenamiento que usan el índice y el lector (`test_calendar_price_windows.py`, `test_window_indices.py`, `test_financial_run.py`, `test_candidate_run.py`, `test_titans_walk_forward.py` y `test_pipeline_batches.py`: 184 superadas y 2 omitidas).

`tests/memory/test_financial_observation_empty_groups.py` sigue comprobando que los grupos vacíos no cambian eventos ni bloques. [`mutations.py`](mutations.py) aplica 10 mutaciones dirigidas sobre una copia del árbol ([`mutations.json`](mutations.json)). Las 10 se detectan. La que deja escribir tablas vacías solo la detecta `test_assets_without_matured_labels_leave_no_empty_piece`, porque `concat_tables` no cambia ninguna fila al reunirlas y solo deja fragmentos vacíos.

## Reproducción

[`index_build.py`](index_build.py) abre una vista, construye los cuatro índices en una carpeta nueva como el primer trabajo de la ventana y guarda el tiempo de cada fase, el pico de memoria del proceso y la huella de cada archivo de eventos. Se ejecuta desde la raíz de cada árbol con `PYTHONPATH=src`, en su propio trabajo `memslot heavy`:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python reports/engineering/observation-index-build-20261014/index_build.py \
  VISTA/manifest.json ÍNDICES salida.json
```
