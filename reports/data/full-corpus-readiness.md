# Cobertura previa a la campaña completa

Comprobación del 22 de septiembre de 2026. El alcance es presencia de fuentes,
ventanas de precios y disponibilidad de los derivados existentes. No es una
validación editorial de todas las noticias ni un entrenamiento.

La [corrección posterior del inventario](chart-identity-repair.md) resuelve la
asignación de gráficos descrita en esta captura, conservando sus fuentes.

La [auditoría contable posterior](full-company-audit.md) recorre los archivos de
los 2.639 candidatos estadounidenses. Su alcance sustituye al recuento del panel
contable pequeño como evidencia de cobertura del universo, sin convertir esos
hechos en muestras multimodales admitidas.

## Fuentes y ventanas

El [inventario completo](full-source-inventory.json) registra 216.453 archivos,
116.185.016.265 bytes y la huella
`f206a659be9e19f2cd6913488843c480c06808ce089c19cef95cd62d672c1619`.
La huella agregada de sus registros coincide con ese informe. No se han releído
los 116 GB para afirmar una nueva verificación íntegra de todos sus contenidos.

Se revisaron los 195.382 nombres de PNG y su carpeta de activo. No quedaron nombres
sin resolver. En 142.468 imágenes estadounidenses y 26.418 chinas, `record.symbol`
no identifica correctamente el activo debido al anidamiento variable de carpetas.
Por ejemplo, `image/image/S&P500_image_a/aa/aa_2000_H1_candlestick.png` aparece
clasificado como `S&P500_IMAGE_A`, aunque la carpeta final y el nombre identifican
`AA`. Los bytes y hashes del archivo no se modificaron.

La intersección se calculó por mercado e identidad reconstruida, no por el campo
defectuoso. La presencia de tres fuentes permite estudiar un gráfico regenerado,
pero no prueba que exista una ventana de 64 sesiones consecutivas.

| Comprobación, en número de instrumentos | Estados Unidos | China |
| --- | ---: | ---: |
| Precios, noticias, fundamentales e imágenes originales presentes | 2.639 | 810 |
| Precios, noticias y fundamentales presentes | 2.639 | 810 |
| Admitidos por el selector técnico que exige sector actual | 2.536 | 790 |
| Al menos una ventana de 64 sesiones de precios válidos | 2.634 | 810 |
| Al menos una ventana que termina hasta 2022 | 2.622 | 810 |
| Al menos una ventana que termina durante 2023 | 2.627 | 810 |

El filtro de sector deja fuera 103 instrumentos estadounidenses y 20 chinos con
las fuentes presentes. Es una regla del panel técnico, no una condición válida
para afirmar que se ha procesado todo el universo admisible. BNKD, BNKU, NRGD,
NRGU y VRM solo tienen 27 filas de precios válidos y no forman una ventana de 64.

La comprobación de ventanas verificó las huellas de 5.023 Parquet de precios,
852.835.869 bytes, y decodificó únicamente `session`, en lotes de 1.024 fechas y
con un archivo abierto cada vez. Se leyeron 20.519.186 fechas. Las columnas de
sesión ocupan 58.605.506 bytes comprimidos según los metadatos. El estado de
precios coincide con el informe detallado y con la instantánea de precios del
inventario. No se consultaron retornos ni etiquetas.

| Ventanas de precios posibles entre instrumentos con las tres fuentes | Hasta 2022 | Durante 2023 |
| --- | ---: | ---: |
| Estados Unidos | 9.470.018 | 651.360 |
| China | 2.472.586 | 195.530 |

Estos son límites de disponibilidad del componente temporal y del gráfico, no
recuentos de muestras con noticias y fundamentales válidos. Los periodos se
cuentan por el final de la ventana, sin exigir que una empresa siga presente en
fechas posteriores. La inspección de sesiones desde 2024 solo forma parte del
inventario y no se utiliza para seleccionar la cohorte de desarrollo.

La lectura final tardó 14,29 segundos y alcanzó 170,29 MiB de RSS. Un intento
anterior limitado a 512 MiB de espacio virtual se detuvo antes de recorrer los
Parquet. Ese fallo no demostró que se hubieran consumido 512 MiB de RAM. La
comprobación final evitó acumular registros completos y midió memoria residente.

## Texto y fundamentales

La [revisión editorial acumulada](news-coverage-expansion.md) acredita 13 artículos
de 36 revisados. Su materialización produjo 65 muestras estrictas: 30 de MNST y
35 de DECK. ABM y CSGS no aportan muestras en esa preparación. No se extrapola la
tasa de admisión de esta selección dirigida al resto del corpus.

La auditoría de 29.957 registros de texto en 22 activos admitía 29.890 mediante
las reglas técnicas iniciales. Esa cifra no acredita correspondencia editorial
de todos sus cuerpos. No se puede usar como sustituto del registro de noticias
completas verificadas.

La [auditoría contable china](china-publication-audit.json) encontró 202.767 registros
sin publicación acreditada. Los 62 activos chinos ya normalizados tienen cero
hechos admitidos. La existencia de 810 instrumentos con archivos de las cuatro
fuentes no resuelve esa carencia temporal.

## Contexto macro calculado

Los hashes de los paneles coinciden con los informes de cálculo [US](macro-US-calculation.json)
y [CN](macro-CN-calculation.json). La cesta tiene 70 originales y 70 fórmulas.
La adquisición contiene 59 series completas, dos fallidas y nueve excluidas.
De las fórmulas, 66 producen algún valor. Hay 125 indicadores observados en alguna
fecha y 15 permanentemente ausentes en estos artefactos.

| Tramo de decisiones | Decisiones | Filas de indicador | Valores observados |
| --- | ---: | ---: | ---: |
| US, hasta 2022 | 5.787 | 810.180 | 558.016 |
| US, 2023 | 250 | 35.000 | 31.232 |
| CN, hasta 2022 | 5.574 | 780.360 | 538.088 |
| CN, 2023 | 242 | 33.880 | 30.235 |

Son decisiones diarias sobre fuentes diarias, semanales, mensuales y trimestrales.
No hay granularidad informativa por minuto o segundo. Los 420 canales numéricos
representan 140 valores transformados, 140 máscaras y 140 antigüedades.

Las ausencias permanentes afectan a petróleo y derivados por fallos del intervalo
de adquisición, índices de estrés y condiciones financieras por falta de versiones
del método, GSCPI por publicación pendiente de verificar y seis candidatos chinos
sin identificadores y vintages suficientes. No se rellenan con cero observado ni
con una reconstrucción retrospectiva no acreditada.

## Variables de empresa calculables

Los 22 activos estadounidenses preparados contienen 38.407 hechos admitidos y
28 conceptos. Sus hashes coinciden con los manifiestos. Es una auditoría del panel
preparado, no de todos los fundamentales originales del universo.

La representación actual selecciona ocho conceptos, pero solo siete tienen
observaciones. `CashAndCashEquivalentsAtCarryingValue` está ausente. La variación
de caja disponible es un flujo y no sustituye el saldo. Tampoco están disponibles
ventas, beneficio neto, deuda financiera o acciones en circulación en estos
derivados, por lo que no justifican márgenes, ROA, ROE, PER o deuda neta.

Para los soportes siguientes se exigieron USD, saldos sin duración, mismo
`period_end`, `filed` y `accession`, ausencia de conflicto y denominador positivo.
Se usó el máximo `available_at` de los componentes. Los hechos desde 2024 se
excluyeron antes de estos recuentos.

| Cociente candidato | Activos / grupos hasta 2022 | Activos / publicaciones de 2023 |
| --- | ---: | ---: |
| Activo corriente / pasivo corriente | 18 / 1.764 | 18 / 148 |
| (Activo corriente − pasivo corriente) / activo total | 18 / 1.761 | 18 / 148 |
| Pasivo total / activo total | 18 / 1.645 | 18 / 150 |
| Patrimonio / activo total | 20 / 1.927 | 19 / 152 |
| Cuentas por cobrar / activo total | 16 / 1.439 | 15 / 122 |
| Cuentas por pagar / activo total | 16 / 1.191 | 13 / 92 |
| Pasivo total / patrimonio positivo | 16 / 1.318 | 14 / 108 |
| Caja / pasivo corriente | 0 / 0 | 0 / 0 |

Son grupos contables compatibles, no decisiones diarias ni ejemplos de entrenamiento.
Los ratios no se han materializado ni evaluado. El cociente con patrimonio positivo
excluye 63 grupos de desarrollo y ocho de 2023. El patrimonio negativo sí puede
conservarse en patrimonio sobre activo cuando el activo sea positivo.

## Qué falta antes de entrenar la campaña

La identidad de las imágenes y el filtro de sector deben corregirse en la ruta de
cobertura completa. Después hay que resolver la intersección temporal con noticias
verificadas y hechos contables, ampliar los derivados empresariales con unidades
y periodos compatibles, y comprobar la continuación operativa de ejecuciones.

El [contrato de campaña](../../docs/engineering/comparison-campaign.md) fija esos
controles y el conjunto de referencias. Ningún recuento de este informe autoriza
a marcar como entrenado un modelo ni a considerar completa la preparación científica.

## Repetir la comprobación de ventanas

La orden reproduce el recuento desde el estado detallado existente. Verifica ese
estado y cada Parquet antes de leer la columna de fechas. No usa retornos,
etiquetas, acceso de red ni escritura en los datos.

```bash
uv run --locked python - <<'PY'
import hashlib
import json
import resource
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from mars_titan.data.temporal import MarketClock

state_path = Path("data/interim/price-detail-state-v2.json")
with state_path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == "b065c8edc3a099dc1f0c8f4ee5e21f3bdf3d897f45f919dbdbd59a2b2e40b42c"
state = json.loads(state_path.read_text())
query = """
SELECT json_extract(record, '$.market'), json_extract(record, '$.symbol'),
       json_extract(record, '$.modality')
FROM files
WHERE present = 1 AND json_extract(record, '$.state') != 'error'
  AND json_extract(record, '$.modality') IN ('prices', 'news', 'fundamentals')
"""
presence = defaultdict(set)
with sqlite3.connect("file:data/interim/source-inventory.sqlite?mode=ro", uri=True) as db:
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA mmap_size=0")
    for market, symbol, modality in db.execute(query):
        presence[market, symbol].add(modality)
positions = {}
for market in ("US", "CN"):
    start = "1990-01-01" if market == "US" else "2000-01-01"
    clock = MarketClock(market, start, "2026-01-01")
    positions[market] = {day.isoformat(): i for i, day in enumerate(clock.days)}
assets, windows = defaultdict(Counter), defaultdict(Counter)
for item in state["files"].values():
    market, symbol = item["market"], item["symbol"]
    path = Path(state["details_root"]) / market / symbol / "prices.parquet"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == item["artifacts"]["prices.parquet"]["sha256"]
    previous, run, counts = None, 0, Counter()
    with pq.ParquetFile(path, memory_map=False) as file:
        for batch in file.iter_batches(columns=["session"], batch_size=1024, use_threads=False):
            for day in batch.column(0).to_pylist():
                position = positions[market][day]
                assert previous is None or position > previous
                run = run + 1 if previous is not None and position == previous + 1 else 1
                previous = position
                if run >= 64:
                    period = "hasta_2022" if day <= "2022-12-31" else (
                        "2023" if day <= "2023-12-31" else "desde_2024"
                    )
                    counts[period] += 1
    if presence[market, symbol] == {"prices", "news", "fundamentals"}:
        assets[market].update({period: 1 for period in counts})
        windows[market].update(counts)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > 512:
        raise MemoryError("Se supera el presupuesto residente de esta comprobación")
print(json.dumps({"activos": assets, "ventanas": windows}, indent=2))
PY
```
