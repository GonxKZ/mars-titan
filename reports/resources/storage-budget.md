# Almacenamiento de los paneles preparados

Medición local del 19 de septiembre de 2026 a las 18:14 UTC. Los 316 archivos
Parquet preparados ocupan **204.693.064 bytes lógicos**
(195,21 MiB) y
205.459.456 bytes asignados. Permiten reutilizar
precios, noticias, hechos contables y representaciones sin reconstruirlos en cada
época. Solo está convertida la selección descrita debajo.

El corpus original conserva 216.453 archivos y
**116.185.016.265 bytes lógicos** (108,21 GiB).
Su espacio asignado es 116.631.822.336 bytes. El conjunto de
rutas, tamaño, fecha de modificación y fecha de cambio coinciden con el inventario.
La huella del inventario sigue siendo
`f206a659be9e19f2cd6913488843c480c06808ce089c19cef95cd62d672c1619`.
Esta comprobación no vuelve a leer los 116 GB para calcular sus hashes. La última
verificación íntegra consta en
[full-source-inventory.json](../data/full-source-inventory.json), a las 15:17 UTC.

## Espacio observado

Bytes lógicos significa `stat.st_size`. Bytes asignados significa
`stat.st_blocks × 512`. La segunda columna de tamaño recoge la contabilidad del
sistema de archivos, sin directorios, metadatos globales ni garantías de espacio
físico único si existen bloques compartidos.

| Capa | Archivos | Bytes lógicos | Bytes asignados |
| --- | ---: | ---: | ---: |
| Originales | 216.453 | 116.185.016.265 | 116.631.822.336 |
| Parquet de precios, noticias y fundamentales | 288 | 79.174.964 | 79.896.576 |
| Parquet de muestras numéricas | 26 | 121.376.995 | 121.417.728 |
| Parquet macro de EE. UU. y China | 2 | 4.141.105 | 4.145.152 |
| Manifiestos preparados | 122 | 3.342.294 | 3.575.808 |
| Caché SQLite de representaciones | 1 | 228.376.576 | 228.380.672 |
| Checkpoints de ensayos existentes | 56 | 39.078.838 | 39.247.872 |
| Inventario SQLite | 1 | 160.100.352 | 160.104.448 |

Las capas derivadas enumeradas, incluido el inventario, suman
635.591.124 bytes lógicos
(606,15 MiB). No incluyen entornos,
pesos de codificadores descargados en las cachés del usuario, descargas macro,
etiquetas temporales ni otros resultados fuera de estas categorías.

La caché es compartida por fase 1 y piloto y se cuenta una vez. Contiene 32.648
vectores de noticias y 32.857 de gráficos, con 117.438.464 bytes de valores
`float32`. La diferencia con el archivo SQLite incluye claves, identidades,
checksums, índices y su estructura interna. El contexto macro y los vectores de
las muestras también aparecen en otras capas como archivos distintos, no como
observaciones nuevas.

## Cobertura real

| Panel | Activos con modalidades preparadas | Activos con muestras | Muestras |
| --- | ---: | ---: | ---: |
| Fase 1, EE. UU. | 22 | 22 | 25.856 |
| Fase 1, China | 62 | 0 | 0 |
| Piloto, EE. UU. | 12 | 4 | 7.001 |

El piloto con muestras comprende ABM, CSGS, DECK y MNST. Los 62 activos chinos
tienen precios y noticias, pero sus fundamentales admitidos están vacíos por falta
de evidencia suficiente sobre su publicación. No hay muestras multimodales chinas
listas para los ensayos.

Los manifiestos preparados referencian 480 archivos originales de precios,
noticias y tablas, con 1.734.344.547 bytes.
La reducción hasta los Parquet combina selección, normalización, deduplicación y
exclusiones temporales. No representa una tasa de compresión del corpus completo.
Todos los manifiestos conservan `training_ready=false`. El uso en ensayos de coste
no fija todavía la cohorte y las particiones de la evaluación definitiva.

## Particiones y memoria

Los Parquet con datos usan Zstandard y grupos de filas de hasta 8.192 registros.
Las muestras guardan listas de tamaño fijo `float32` para noticias, gráficos,
fundamentales y macro. Las ventanas de precios de 64 sesiones se calculan al
consumir cada muestra y no se duplican en disco.

El lector proyecta solo las columnas necesarias y entrega bloques de hasta 256
muestras, sin hilos internos de lectura. Mantiene la historia OHLCV del activo
actual. La mayor historia preparada observada tiene 6.348 filas. Su matriz NumPy
de cinco columnas `float64` necesita exactamente **253.920 bytes de valores** por
lector activo. Esta cifra no incluye las tablas Arrow ni las copias transitorias.

| Valores de entrada | Dimensión por muestra | Bytes por muestra | Bytes en lote de 64 |
| --- | ---: | ---: | ---: |
| Noticias | 384 | 1.536 | 98.304 |
| Gráficos | 512 | 2.048 | 131.072 |
| Fundamentales | 24 | 96 | 6.144 |
| Macro | 420 | 1.680 | 107.520 |
| Ventana OHLCV | 64 × 5 | 1.280 | 81.920 |
| Total de entradas `float32` | 1.660 | 6.640 | 424.960 |

El objetivo escalar `float32` añade 256 bytes por lote de 64. Entradas y objetivo
suman **425.216 bytes de valores**. En el bloque de lectura de 256 filas, los cuatro
vectores materializados suman 1.372.160 bytes. El índice de precio y el instante
de predicción añaden 4.096 bytes. Son tamaños calculados por dimensión y tipo,
no una medición del consumo total de RAM o VRAM.

El pico del proceso también incluye bitmaps, objetos Python, buffers, copias,
modelo, activaciones, gradientes, optimizador y reservas del asignador. No se
puede deducir de los bytes del lote. Las mediciones de entrenamiento disponibles
están en [campaign-budget.md](campaign-budget.md) y sus registros enlazados.
Esta auditoría no ejecuta otro benchmark ni establece un rendimiento máximo.

El mayor conjunto original de precios, noticias y tablas por activo corresponde
a HURC, con 132.199.783 bytes,
de los cuales 131.248.632 son tablas. Le siguen VIRC, con 128.032.404 bytes, y
HTLD, con 126.727.174. Se han agregado los archivos mediante el inventario, sin
cargar sus contenidos en memoria.

| Partición preparada | Bytes lógicos | Bytes asignados |
| --- | ---: | ---: |
| Fase 1, ESGRO, tres modalidades | 12.933.907 | 12.939.264 |
| Fase 1, CMG, tres modalidades | 8.785.212 | 8.794.112 |
| Fase 1, AMGN, tres modalidades | 7.934.907 | 7.938.048 |
| Fase 1, AMGN, muestras | 11.976.151 | 11.976.704 |
| Fase 1, CMG, muestras | 11.356.492 | 11.358.208 |

`ESGRO/news.parquet` es el mayor archivo normalizado, con 12.856.172 bytes y
8.575 noticias. Sus columnas suman 41.151.044 bytes sin comprimir según los
metadatos Parquet. Esta cifra tampoco equivale a su representación en objetos
Python ni al pico de RAM.

La conversión aún acumula las noticias, rechazos y hechos deduplicados de cada
activo. La materialización acumula sus muestras y carga el panel macro completo.
Para convertir el resto del universo deben aplicarse límites comprobados por
activo y fragmentación o lectura y escritura incremental para los casos que los
superen. La lectura por lotes de entrenamiento no resuelve por sí sola ese coste
de preparación.

## Repetición de la medición

Desde la raíz del repositorio, este comando reproduce los tamaños por capa,
contrasta los metadatos originales y calcula la misma huella a partir de SQLite.
No modifica datos ni deserializa tensores.

```bash
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

totals = defaultdict(lambda: [0, 0, 0])
def record(group, path):
    info = path.stat()
    sizes = totals[group]
    sizes[0] += 1
    sizes[1] += info.st_size
    sizes[2] += info.st_blocks * 512
    return info

digest = hashlib.sha256()
with sqlite3.connect("file:data/interim/source-inventory.sqlite?mode=ro", uri=True) as db:
    root = Path(db.execute("SELECT value FROM metadata WHERE key='root'").fetchone()[0])
    for size, mtime, ctime, raw in db.execute(
        "SELECT size,mtime,ctime,record FROM files WHERE present=1 ORDER BY path"
    ):
        row = json.loads(raw)
        info = record("originals", root / row["path"])
        assert (info.st_size, info.st_mtime_ns, info.st_ctime_ns) == (size, mtime, ctime)
        digest.update(json.dumps(
            [row["path"], row["sha256"], row["bytes"]],
            ensure_ascii=True, separators=(",", ":")
        ).encode("ascii") + b"\n")
for root in ("data/processed/phase1", "data/processed/pilot",
             "data/embeddings", "data/interim/budget-training"):
    paths = subprocess.check_output(
        ["rg", "--files", "--hidden", "--no-ignore", "--null", root]
    )
    for raw in paths.split(b"\0"):
        if not raw:
            continue
        path = Path(os.fsdecode(raw))
        if root == "data/embeddings":
            group = "embedding_cache"
        elif root == "data/interim/budget-training":
            if path.suffix != ".pt":
                continue
            group = "budget_checkpoints"
        elif path.name == "manifest.json":
            group = "prepared_manifests"
        elif path.suffix == ".parquet":
            group = ("sample_features" if "samples" in path.parts else
                     "macro" if path.name.startswith("macro-") else "prepared_modalities")
        else:
            continue
        record(group, path)
record("inventory_database", Path("data/interim/source-inventory.sqlite"))
print(json.dumps({"columns": ["files", "logical_bytes", "allocated_bytes"],
                  "groups": dict(totals), "snapshot_sha256": digest.hexdigest()}, indent=2))
PY
```

Para inspeccionar una partición sin leer sus filas:

```bash
uv run --no-sync python - <<'PY'
import pyarrow.parquet as pq

path = "data/processed/phase1/samples/US/AMGN/samples.parquet"
with pq.ParquetFile(path) as parquet:
    print(parquet.schema_arrow)
    print(parquet.metadata)
    for index in range(parquet.metadata.num_row_groups):
        print(parquet.metadata.row_group(index))
PY
```

El [registro JSON](storage-budget.json) conserva tamaños por mercado y modalidad,
símbolos de los paneles, las mayores particiones, la caché y los tres directorios
de checkpoints. No se han modificado los originales, los Parquet ni los checkpoints.
