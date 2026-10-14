# Catálogo de la edición y subconjuntos declarados

Tarea [#537](https://github.com/GonxKZ/mars-titan/issues/537). La edición v3.1 guarda cada activo en su propio Parquet y sus objetivos se construyen aparte. Antes de decidir cuántas filas, épocas y etapas tendrá el entrenamiento hace falta saber qué hay en cada muestra y poder elegir subconjuntos reproducibles sin copiar datos, con las mismas filas para todos los modelos. Este documento describe el catálogo, los subconjuntos, su paso por las vistas y la campaña y el dimensionado. Los recibos con datos reales están en [el informe del 13 de octubre](../../reports/data/edition-catalog-20261013/README.md). No se ha entrenado ni evaluado ningún modelo.

## Catálogo

`training/edition_catalog.py` escribe `catalog.parquet` y su `manifest.json` en una carpeta nueva que se publica de una vez. Hay una fila por muestra de la edición, en el orden de los activos del manifiesto de objetivos y de las muestras de cada archivo. La clave es (`market`, `asset_id`, `prediction_at`), la misma que usan las predicciones y `integrity.row_identity`.

| Columnas | Contenido |
| --- | --- |
| `market`, `asset_id`, `session`, `prediction_at` | Identidad de la muestra y sesión del mercado de su decisión |
| `<bloque>_available_at` | Disponibilidad declarada de precios, noticias, gráficos, fundamentales y macro |
| `inputs_available_at` | Máximo de esas disponibilidades y de la fila de precios que cierra la ventana, igual que lo calcula la lectura de las vistas |
| `has_<bloque>`, `news_count` | Bits de presencia de la máscara y número de noticias admitidas |
| `label_reason`, `annual_boundary_label` | Motivo del objetivo y si es la etiqueta del corte anual que las vistas recuperan |
| `target_partition`, `target_available_at` | Partición de los objetivos y madurez de la etiqueta |
| `samples_file`, `sample_row`, `row_group`, `row_in_group`, `label_row` | Archivo, posición y grupo de filas de origen de la muestra y de su etiqueta |

No contiene vectores ni el valor del objetivo. Las columnas de entrada nunca son posteriores a `prediction_at`, y la construcción se detiene si una lo fuera. La madurez de la etiqueta sí es posterior por definición, pero solo dice cuándo se conoce la etiqueta. Elegir filas por ella no usa su valor. El manifiesto guarda las huellas de la edición, de los objetivos y del descriptor de la representación, los recuentos por mercado, año, motivo, partición y bloque, y la huella de cada archivo de muestras, etiquetas y precios.

**Test reservado.** Antes de leer un archivo de muestras o de etiquetas se comprueban las estadísticas de `prediction_at` de cada grupo de filas. Si alguno llega a 2024, la construcción se detiene sin leer sus filas. Un archivo sin estadísticas también la detiene, porque no permitiría comprobarlo.

**Memoria acotada.** De cada archivo solo se leen las columnas de metadatos, con Arrow y sin hilos. Los activos se acumulan hasta completar grupos de 2²⁰ filas, que se escriben con zstd. La memoria depende del activo mayor y del tamaño del grupo, no de la edición. Los recibos guardan tiempo, memoria residente máxima y bytes en espera.

**Verificación.** `verify_catalog` no usa el constructor ni la lectura de entrenamiento. Vuelve a calcular la huella de cada archivo de muestras y de etiquetas, los lee con Arrow y compara fila a fila clave, presencia, motivo, partición, madurez y enlace de cada etiqueta. Exige además las particiones y el número de muestras de los objetivos. La misma edición y los mismos objetivos producen un `catalog.parquet` idéntico bit a bit.

```bash
uv run python -m mars_titan.training.edition_catalog build \
  --targets <targets-v3.1>/manifest.json --edition <encoded-history-v3.1>/manifest.json \
  --output <catálogo>
uv run python -m mars_titan.training.edition_catalog verify \
  --catalog <catálogo>/manifest.json --report <informe>.json
```

**Catálogo de la v3.1.** Sobre la edición `a459dbb8…` y los objetivos `fe86a923…`, el catálogo tiene 18.698.976 filas de 5.008 activos (15.874.289 de US desde 2000 y 2.824.687 de CN desde 2006), ocupa 257 MB y su archivo tiene la huella `6edfddf9…`. Lo construyó un solo proceso en 3 min 19 s con 1,73 GiB de memoria residente máxima. La verificación independiente releyó los 59 GB de muestras y confirmó las 18.698.976 filas y las particiones de los objetivos (17.142.187 de `train` y 1.235.785 de `validation`) en 36 min 37 s, limitada por el disco compartido con otras sesiones.

## Subconjuntos declarados

`training/edition_subsets.py` elige filas del catálogo con una definición JSON versionada. Las reglas se aplican en este orden y ninguna mira un objetivo:

| Regla | Efecto |
| --- | --- |
| `markets` | Mercados incluidos, ordenados |
| `years` | Primer y último año de la sesión, entre 2000 y 2023 |
| `labels` | `usable` conserva las filas cuyo objetivo puede entrar en una vista (aceptado o recuperado en el corte anual) y `all` todas |
| `modalities.required` | Bloques que deben estar presentes |
| `liquidity` | Universo de liquidez de cada año, decidido en su corte |
| `max_per_asset` | Máximo de filas por activo |
| `sample` | Fracción por estrato (`market`, `year`, `target_partition`) con semilla |

El máximo por activo y el muestreo conservan las filas con menor clave de orden `row_order_v1`. La clave es una mezcla splitmix64 de la semilla, de una huella BLAKE2b del activo y del instante de la predicción. No depende del orden de lectura ni de las demás filas, así que la misma definición sobre el mismo catálogo da los mismos identificadores aunque cambien los grupos de filas. Cada estrato conserva `floor(fracción · n + 1/2)` filas.

`materialize` escribe `ids.parquet` con las claves ordenadas y un manifiesto con la definición, su huella, la del catálogo y la de los identificadores. `verify_subset` vuelve a seleccionar desde el catálogo y exige los mismos identificadores.

**Lectura y memoria.** El catálogo se lee por grupos de filas y solo con las columnas de la selección. Los filtros por fila se aplican con Arrow antes de pasar nada a numpy. De cada fila que pasa se guardan su instante y tres códigos compactos (activo en 32 bits, año en 16 y partición en 32). El mercado y la huella de cada activo se guardan una vez por activo. Los estratos y los pares de activo y año son claves enteras en un rango pequeño, así que se agrupan marcando y contando en vez de ordenar las filas. Los grupos se ordenan en el tipo sin signo más pequeño que los contiene, que numpy ordena por radix. Ninguno de estos cambios altera la selección, y los identificadores de la v3.1 salen idénticos byte a byte a los de la primera versión.

## Universo de liquidez

La regla `liquidity` reutiliza la asignación de [los estratos de liquidez](../research/metrics.md#estratos-de-liquidez-y-métricas-cuadráticas) de #532 con la edición sin ajustar. Se declara con su fecha, la fuente, la identidad de la edición sin ajustar, el corte, los umbrales de precio y de volumen, la ventana del volumen y los estratos que se conservan:

```json
"liquidity": {
  "declared_at": "2026-10-13",
  "source": "unadjusted_reconstructed_verified_rows_up_to_the_last_published_session_v1",
  "edition_id": "1ac37278…",
  "cut": "calendar_year_start_utc_v1",
  "price_below": 1.0,
  "volume_below": 1000,
  "volume_sessions": 20,
  "keep": ["liquid"]
}
```

El corte de un año es el 1 de enero a las 00:00 UTC. En ese instante cada activo recibe su estrato con la última sesión publicada, que es la última del año anterior: el último cierre negociado verificado y la mediana del volumen de las 20 filas que terminan en ella. Las filas cuya sesión cae en ese año se conservan si el estrato del activo está en `keep`. El universo de un año solo depende de sesiones anteriores al año, y por tanto anteriores a la predicción de cada fila y a cualquier corte del walk-forward que empiece ese año o después. En los protocolos v2 de US y CN, cada ventana de evaluación empieza el 1 de enero (la validación empieza el 1 de abril y dura seis meses, seguida de tres de calibración), así que cada una se evalúa sobre el universo fijado justo antes de empezar.

Un activo sin sesiones verificadas suficientes antes del corte queda `unclassified` y solo se conserva si se declara. Por eso el primer año de cada mercado (2000 en US y 2006 en CN) queda fuera de un universo líquido, porque la edición sin ajustar no tiene sesiones anteriores. `materialize` necesita la carpeta de la edición sin ajustar (`--liquidity-edition`), comprueba que su identidad es la declarada y guarda en el manifiesto su ruta y cuántos activos caen en cada estrato en cada corte. `verify_subset` vuelve a asignar los estratos con esa edición y exige el mismo universo.

## Subconjuntos de la v3.1

`configs/targets/edition-subsets/` declara tres definiciones. Se materializaron y verificaron sobre el catálogo de la v3.1:

| Definición | Filas | Activos | US | CN | Horas neuronales frente al diseño completo |
| --- | ---: | ---: | ---: | ---: | ---: |
| `usable-2000-2023` | 18.382.961 | 5.005 | 15.609.969 | 2.772.992 | 1 |
| `liquid-usable-2000-2023` | 15.724.204 | 4.366 | 13.253.850 | 2.470.354 | 0,854 |
| `liquid-usable-half-2000-2023` | 7.862.112 | 4.366 | 6.626.931 | 1.235.181 | 0,427 |

`usable-2000-2023` reproduce las 18.377.972 etiquetas aceptadas más las 4.989 recuperadas en el corte anual. En US hay 64.250 combinaciones de activo y año entre 2000 y 2023. En su corte, 52.897 son líquidas, 1.196 tienen el precio por debajo de 1 USD, 1.825 negocian menos de 1.000 títulos, 125 cumplen las dos condiciones y 8.207 quedan sin clasificar, 1.311 de ellas en 2000. En CN hay 11.913: 10.284 líquidas, 234 con poco volumen y 1.395 sin clasificar, 444 de ellas en 2006. Las horas son el cociente del dimensionado de la campaña A v2 con el mismo caudal uniforme para el subconjunto y para el diseño completo. Las horas absolutas esperan a los caudales medidos con las vistas de la v3.1.

## Vistas y campaña

Una campaña v2 puede declarar `data_subset` con la ruta y la huella del manifiesto del subconjunto. `load_campaign` comprueba la huella y guarda su identidad (definición, catálogo, identificadores y filas). Al preparar las vistas de la campaña, `prepare_temporal_corpus` y la unión conjunta reciben el subconjunto:

- comprueban que el catálogo describe los mismos archivos de muestras y etiquetas que el padre de la vista,
- excluyen con el motivo `outside_subset` las etiquetas usables que no pertenecen al subconjunto, sin cambiar su motivo de origen,
- guardan la identidad del subconjunto en el manifiesto de cada ventana y en el informe.

Las filas son independientes al asignar tramos (`temporal_corpus.fold_rows`), así que excluir filas no cambia el tramo de las demás. El subconjunto filtra etiquetas y no observaciones. Las familias cronológicas leen la vista con el índice de `memory/financial_observations.py`, que recorre todas las decisiones de la ventana tengan o no etiqueta, y `observation_batches` hace lo mismo. Filtrar también las observaciones con `labels: usable` haría que el recorrido dependiera de que exista la sesión siguiente de cada activo (`missing_next_session`), algo que no se conoce al decidir. Al leer la vista se comprueba que su subconjunto sigue siendo el declarado. `scope_views` rechaza vistas cuyo subconjunto no coincida con el de la campaña, y la identidad de cada trabajo cambia con la huella de su vista.

## Dimensionado

`campaign_budget` admite `--catalog`. El recuento recorre el catálogo con la misma asignación por ventana que publica las vistas, con la disponibilidad de las entradas y la recuperación del corte anual, y coincide con los recuentos de sus manifiestos. También cuenta las filas nuevas de cada ventana con padre del walk-forward por etapas. Con `--subset`, o con el subconjunto que declare la campaña, proyecta las horas del subconjunto y del diseño completo con los mismos caudales y épocas y da su cociente. Una campaña con subconjunto no se dimensiona con `--targets`, que cuenta todas las filas, y unos recuentos de `--counts` solo sirven si llevan su mismo subconjunto. Los recuentos escritos desde el catálogo o desde las vistas guardan esa identidad. Los cortes de cada ventana (`FoldPartitioner`) se preparan una vez por ventana con `temporal_corpus.fold_partitioners` y se reutilizan en todos los activos, igual que al preparar las vistas. Antes se rehacían para cada activo y ventana, lo que convertía de nuevo el calendario completo de decisiones en cada llamada.

```bash
uv run python -m mars_titan.training.campaign_budget \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --catalog <catálogo>/manifest.json --subset <subconjunto>/manifest.json \
  --throughput <caudales medidos>.json
```

## Comprobaciones

- `tests/training/test_edition_catalog.py`: una fila por muestra alineada con muestras y etiquetas, posiciones de origen, disponibilidad de las entradas con la fila de precios que cierra la ventana, reconstrucción idéntica con otros grupos de filas, mutaciones de filas detectadas por la verificación, parada sin leer filas de 2024 y parada ante una entrada futura o una edición ajena.
- `tests/training/test_edition_subsets.py`: filtros frente a una lectura directa, máximo por activo y fracción por estrato frente a su definición, independencia de los grupos de filas, clave de orden frente a una implementación de splitmix64 con enteros de Python y sin modificar sus entradas, primeras filas de cada grupo frente a una ordenación directa con más de 255 y de 65.535 grupos, recuento de presencia frente a `np.unique`, huellas protegidas, definiciones inválidas y las definiciones del repositorio.
- `tests/training/test_edition_subset_liquidity.py`: universo de cada año igual al estrato calculado aparte con las filas de la edición sin ajustar anteriores al corte, un activo que se vuelve ilíquido durante el año que conserva sus filas de ese año, el mismo resultado con sesiones posteriores cambiadas, `unclassified` solo cuando se declara, máximo por activo después del universo, identidad de la edición y universo publicado comprobados al verificar.
- `tests/training/test_edition_subset_views.py`: recuentos del catálogo iguales a los de las vistas publicadas con y sin subconjunto, filas nuevas iguales a las que cuenta `campaign_budget` desde los objetivos, observaciones iguales con y sin subconjunto, campaña y vistas con el mismo subconjunto, comparación con el diseño completo y rechazo de los recuentos de otro diseño.
- [Mutación dirigida](../../reports/data/edition-catalog-20261013/mutations.json) sobre la selección, el corte de liquidez, el catálogo, las vistas, la observación, la campaña y el dimensionado: las 38 mutaciones hacen fallar alguna prueba.

El 14 de octubre, después de liberar espacio en disco, se repitieron `edition_catalog verify` y `edition_subsets verify` sobre los artefactos de disco. Las tres verificaciones de subconjuntos y la del catálogo terminaron con `verified`, con las mismas huellas y recuentos ([recibo](../../reports/data/campaign-a-v2-data-review-20261014.json)).

## Límites

- El estado de verificación de la edición sin ajustar usó acciones corporativas posteriores para deshacer el ajuste del proveedor, como se explica en los estratos de liquidez. El valor de cada cierre ya era público en su sesión, pero qué activos quedan sin clasificar depende de esa reconstrucción. Todos los modelos usan las mismas filas, así que esa selección no favorece a ninguno.
- El universo de liquidez cambia una vez al año. Un activo que se vuelve ilíquido a mitad de año conserva sus filas hasta el corte siguiente.
- El catálogo no copia vectores, así que un subconjunto no acelera la lectura de los archivos de muestras de un activo, solo reduce las filas que se ajustan, se seleccionan y se evalúan. Las familias cronológicas siguen observando todas las decisiones de cada ventana.
- La selección carga en memoria las claves de las filas que pasan los filtros. Con la v3.1 completa, materializar o verificar un subconjunto llega a 2,2 GiB de memoria residente, frente a 3,8 GiB de la primera versión.
- Con liquidez, la mayor parte del tiempo es la asignación de estratos de `evaluation.liquidity_strata`, que lee el Parquet de precios de cada activo de la edición sin ajustar. No se ha cambiado porque su código forma parte de las huellas de los informes de evaluación.
