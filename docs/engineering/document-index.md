# Índice documental anterior a la agregación

El índice lateral conserva las menciones de los Parquet de noticias preparados por `write_cohort_news`. No modifica esos archivos ni necesita un codificador. Reutiliza las representaciones de `EmbeddingCache` mediante la identidad ya utilizada por `_text_window`: codificador, tipo `news`, contenido y regla de disponibilidad. Si falta una representación compatible, la preparación falla.

`prepare_document_index` recibe un diccionario de activo a manifiesto editorial, una caché SQLite y la huella del codificador. Comprueba la cohorte, el mercado, el símbolo, las huellas de los archivos y el corte anterior a 2024. Las fuentes de distinta cohorte no pueden combinarse. Publica `documents.parquet` y `manifest.json` juntos en un directorio nuevo. Una interrupción anterior a la publicación no deja una edición utilizable. Repetir la preparación reutiliza una edición confirmada de la misma configuración.

## Identidades y procedencia

El identificador de documento combina contenido, URL y publicación declarada. El identificador de mención combina activo e `event_id`. No son identificadores universales de acontecimientos. Dos activos pueden conservar menciones distintas del mismo documento y compartir su representación. Una mención repetida con metadatos incompatibles se rechaza. Una repetición exacta se cuenta y se conserva una sola vez.

Cada fila conserva `content_hash`, `event_id`, `source_file`, `source_record_hash`, `line`, disponibilidad, regla aplicada, publicación y campos de revisión editorial. El cuerpo permanece en el Parquet original. El manifiesto identifica su archivo y su huella. El índice no vuelve a copiar los cuerpos. Los campos de revisión no se convierten en una afirmación nueva de validez histórica.

`embedding_key` identifica la fila de caché y `embedding_sha256` fija los bytes del vector. La lectura es de solo lectura y verifica identidad, dimensión, finitud y checksum. Una sustitución de vector bajo la misma clave también se detecta. Las entradas se comprueban al consumirlas, por lo que reutilizar el índice no certifica de antemano toda una caché que pueda cambiar.

## Ventanas, límites y recuperación

`DocumentIndex.batches(asset_id, start, end)` usa límites inclusivos y fechas con zona horaria. Reutiliza `NewsWindows` con dos grupos Arrow como máximo. El orden es disponibilidad, contenido, evento y activo. El lector no entrega documentos posteriores al corte y rechaza consultas que abran la reserva final.

El cursor incluye la identidad de índice y consulta, el consumo confirmado y la huella del prefijo. Para recuperar se vuelve a recorrer ese prefijo y se comprueba su huella. Esto cuesta lectura repetida, pero no requiere retener todos los documentos ni depende del estado de un iterador anterior.

Los límites predeterminados son 16 MiB por grupo Parquet, 64 menciones por lote, 8 MiB de tamaño serializado estimado por lote, 100.000 menciones seleccionables y un millón de filas examinadas por consulta. La preparación ordena mediante SQLite con una caché de 8 MiB y un máximo de 1 GiB para su archivo. Los metadatos Parquet se limitan a 8 MiB. Hay límites adicionales de filas, grupos y dimensiones. Estos valores acotan datos y estructuras de trabajo, no el RSS total del intérprete y sus bibliotecas.

La memoria de la comparación depende de dos grupos, el lote, hasta 1.024 vectores seleccionados y dos conjuntos de identidades acotados por el máximo de menciones. La consulta temporal examina también las filas de otros activos dentro de la ventana. El límite de lectura incluye esas filas. El índice todavía no está particionado físicamente por activo.

## Comparación ejecutable

El control calcula la media de todas las menciones disponibles y la media de las últimas `k`, con desempate por el orden del índice. Ambas reciben exactamente la misma ventana. Informa de menciones examinadas, documentos distintos, representaciones distintas y lecturas de caché. No entrena un selector ni compara calidad predictiva. No hay ahorro de adquisición atribuido a seleccionar después de preparar todos los embeddings.

Este ejemplo genera fuentes sintéticas separadas, vectores constantes de control y sus manifiestos. No utiliza documentos reales ni presenta sus fechas como evidencia histórica:

```bash
PYTHONPATH=src uv run --no-sync python scripts/example_document_index.py \
  --output /tmp/document-control --assets 8 --documents 128 --repeats 5

PYTHONPATH=src uv run --no-sync python -m mars_titan.data.document_cli compare \
  --index /tmp/document-control/index/manifest.json \
  --cache /tmp/document-control/synthetic-vectors.sqlite \
  --asset US/CONTROL000 --start 2023-07-05T12:00:00+00:00 \
  --end 2023-07-05T14:08:00+00:00 --selected 8 --repeats 5
```

La orden `prepare` acepta `--source ACTIVO=MANIFIESTO` repetido, `--cache`, `--encoder` y `--output`. La API devuelve el manifiesto confirmado. `compare` devuelve JSON y puede escribirlo con `--output` en una ruta nueva.

## Comprobaciones y medidas

El 5 de octubre de 2026 pasaron 52 pruebas de índice, CLI, caché, ventanas, noticias y materialización de cohortes. Cubren igualdad exacta con `_text_window`, invariancia al sufijo futuro, menciones entre activos, recuperación, interrupción antes de publicar, límites, corrupción y representaciones incompatibles. El control de vista completa no atribuye significado financiero a los vectores sintéticos.

Se ejecutaron tres cargas con vectores de 384 componentes en un AMD Ryzen 9 8945HS, Linux x86_64, Python 3.12.14, NumPy 2.5.3 y PyArrow 25.0.1. Cada consulta usó un calentamiento y cinco repeticiones. Había otras tareas locales en ejecución. No se detuvieron para medir.

| Activos × documentos | Filas del índice | Parquet, bytes | Consulta p50, ms | Consulta p95, ms | Menciones del activo/s | Proceso completo, s | Pico del proceso, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 × 8 | 16 | 13.396 | 3,643 | 4,110 | 2.586,24 | 1,39 | 140,38 |
| 8 × 128 | 1.024 | 195.624 | 137,772 | 153,724 | 909,45 | 3,16 | 159,22 |
| 64 × 128 | 8.192 | 1.413.491 | 450,843 | 459,201 | 283,38 | 7,14 | 178,71 |

Los tiempos de consulta incluyen lectura, verificación de caché, media y selección. El proceso completo, medido con `/usr/bin/time`, incluye generación del control, preparación y consultas. El RSS es el máximo del proceso, no un pico atribuible exclusivamente al lector. El ejemplo escribe tiempos individuales, versiones y formas en `comparison.json` y `control.json`. No se midieron energía ni coste económico y no se utilizó GPU. Estas cargas comprueban el recorrido de preparación, no permiten extrapolar calidad ni velocidad sobre todo el corpus.
