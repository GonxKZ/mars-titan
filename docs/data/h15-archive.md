# Ediciones documentales H.15

`macro_h15_documents.extract_h15_release` extrae candidatos de una edición HTML semanal conservada. `macro_h15_archive.prepare_h15_archive` exige además el PDF enlazado, los recibos de captura y una revisión explícita de sus celdas. Publica una edición documental separada con `admission_required=True` y `training_ready=False`. No descarga fuentes ni produce un panel de 140 indicadores.

Se admiten inicialmente las filas nominales Treasury de vencimiento constante de tres meses, dos, cinco, diez y treinta años, junto al tipo efectivo de fondos federales. Las letras a descuento, las subastas y otros tipos se distinguen por sección. Los cinco encabezados diarios deben identificar año, mes y día, estar ordenados y no superar la publicación. Las tres columnas agregadas no se convierten en días. El día de publicación puede ser un martes, como en la edición de 18 de enero de 2000.

## Procedencia y revisión

El manifiesto tiene `schema_version=1` y declara estas reglas:

- `policy`: `FED_H15_REVIEWED_WEEKLY_ARCHIVE_V1`.
- `availability_rule`: `SOURCE_DAY_END_TARGET_NEXT_SESSION_V1`.
- `review_policy`: `PDF_HTML_REVIEWED_CELLS_V1`.

Cada elemento de `documents` contiene `publication_date`, `html`, `pdf` y `review`. Las representaciones HTML y PDF requieren `path`, `sha256`, `url` y `receipt`, este último con su propia ruta y SHA. Las rutas son relativas al directorio del manifiesto y no pueden escapar de él ni atravesar enlaces simbólicos. Las URL deben corresponder a la misma edición HTTPS de `www.federalreserve.gov/releases/h15/YYYYMMDD/`. El enlace PDF se comprueba en el HTML.

Los recibos deben confirmar HTTP 200, cuerpo completo, URL final idéntica, hash y tamaño. El recibo HTML conserva `body_sha256` y `body_bytes`. El del PDF conserva `sha256`, `bytes` y `valid_pdf=True`. La revisión se identifica mediante `path` y `sha256`.

La revisión tiene `schema_version=1`, la política indicada, estado `document_reviewed_not_admitted`, fecha de publicación, hashes de ambas representaciones, unidad `percent_per_annum`, `publication_date_page`, `reviewed_pages` y `correction_notice=False`. Sus treinta elementos de `observations` contienen `indicator_id`, `period_start`, `value_exact`, `source_page`, `source_column` y `source_label`. La columna es diaria y empieza en uno. La etiqueta corresponde a la fila revisada, por ejemplo `10-year` o `Federal funds (effective)`.

El productor compara esa transcripción con las celdas HTML. No reemplaza la revisión visual del PDF ni convierte una coincidencia textual en aprobación. La revisión debe referirse a los bytes cuyo hash declara. El contrato inicial pone fuera de alcance documentos con un aviso de corrección sin resolver. Tampoco permite atribuir un mismo PDF a publicaciones de fechas distintas.

`value_exact` conserva el decimal publicado como texto. La columna `value` ofrece su representación binaria de 64 bits. Un cero observado permanece como cero. Las celdas vacías, `n.a.` y `ND` conservan ausencia con `missing_source_value`, sin generar un valor. Las unidades no se convierten.

## Disponibilidad

La fecha de observación y la fecha de edición se guardan por separado. Se permite historia anterior a 2000, por ejemplo para antecedentes de ese inicio. La publicación debe quedar dentro del corte autorizado y antes de 2024.

Se reutiliza la regla conservadora de `macro._available`: fin del día declarado en `America/New_York`, conversión a la fecha del mercado receptor y siguiente sesión de su `MarketClock`. La regla no toma simplemente el primer cierre posterior. Para la edición del 10 de enero de 2000 produce el 11 a las 21:05 UTC en US y el 12 a las 07:05 UTC en CN. Si no hay una sesión dentro del corte, la edición conserva `available_at=None` y contabiliza esa falta de disponibilidad.

La precisión documental es de día. `CreationDate`, `ModDate`, la fecha de captura y el horario actual de H.15 no se emplean como hora histórica de publicación. Los PDF contrastados no acreditan por sí solos una captura contemporánea de 2000 ni la ausencia de otras revisiones.

## Publicación y recuperación

La llamada mínima es:

```python
from mars_titan.data.macro_h15_archive import prepare_h15_archive

report = prepare_h15_archive(
    "data/external/h15-reviewed/source-manifest.json",
    "data/processed/h15-document-edition",
    markets=("US", "CN"),
    cutoff="2023-12-31",
)
```

Las rutas del ejemplo son entradas y destino que debe proporcionar quien prepare la edición. La CLI equivalente es `python -m mars_titan.data.macro_h15_archive`, con `--manifest`, `--output`, `--cutoff` y `--market` repetible.

La salida contiene `observations.parquet`, `source-manifest.json`, `configuration.json` y `report.json`. Cada boletín aporta treinta observaciones documentales y treinta filas por mercado. Dos boletines y dos mercados producen sesenta observaciones y ciento veinte filas, no ciento veinte hechos independientes.

La identidad enlaza fuentes, revisión, reglas, código y calendarios. La publicación usa un directorio temporal y renombrado sin reemplazo. Al repetir la llamada se verifican otra vez las fuentes y se contrastan el esquema, los valores, la disponibilidad, los hashes y los recuentos de la salida. Un archivo confirmado alterado se rechaza y no se reconstruye en silencio. Los archivos correctos no se reescriben. Una interrupción previa a la confirmación permite repetir la preparación completa desde las fuentes inmutables.

El presupuesto permite hasta 2.048 ediciones, ocho MiB por HTML o PDF, dos MiB por manifiesto o revisión y 512 MiB acumulados de entradas. La lectura DOM limita elementos y celdas. La salida lógica y el Parquet que se recupera tienen un límite de 64 MiB. Se procesan las representaciones de cada documento por separado, sin mantener todos los originales en memoria.

## Comprobación realizada

La integración offline de las ediciones oficiales del [10 de enero](https://www.federalreserve.gov/releases/h15/20000110/h15.pdf) y [18 de enero](https://www.federalreserve.gov/releases/h15/20000118/h15.pdf) reproduce los sesenta decimales revisados y las ciento veinte disponibilidades de US y CN. La recuperación conserva hashes y fechas de modificación. El [recibo](../../reports/data/h15-document-edition-20261007.json) recoge medidas y calidad, con sus límites.

La edición no completa la historia desde 2000. DFF necesita los días de calendario que estos boletines no exponen individualmente y las transformaciones a veintiuna observaciones requieren más historia. La integración con un catálogo macro y su admisión siguen siendo pasos distintos de esta preparación documental. No se han ejecutado modelos.
