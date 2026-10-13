# Catálogo de la edición v3.1 y subconjuntos declarados

Recibos del 13 de octubre de 2026 para #537. No se ha ajustado ni evaluado ningún modelo y no se ha ejecutado ningún paso de optimizador. Los datos de la edición y `dataset/` solo se han leído. El diseño está en [el catálogo de la edición](../../../docs/data/edition-catalog.md).

## Datos reales

[`real-data.json`](real-data.json) reúne las huellas de las fuentes, del código y de cada salida. Todo se ejecutó en CPU con un solo proceso de Python, dos hilos y memslot, en un equipo compartido con otras sesiones.

```bash
M=<history-2000-20261007>/materialization
uv run python -m mars_titan.training.edition_catalog build \
  --targets $M/targets-v3.1/manifest.json --edition $M/encoded-history-v3.1/manifest.json \
  --output <catálogo>
uv run python -m mars_titan.training.edition_catalog verify \
  --catalog <catálogo>/manifest.json --report <informe>.json
uv run python -m mars_titan.training.edition_subsets materialize --catalog <catálogo>/manifest.json \
  --definition configs/targets/edition-subsets/liquid-usable-2000-2023.json \
  --liquidity-edition <unadjusted-prices-20261009>/edition-v1 --output <subconjunto>
uv run python -m mars_titan.training.edition_subsets verify --subset <subconjunto>/manifest.json
uv run python -m mars_titan.training.campaign_budget \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --catalog <catálogo>/manifest.json --subset <subconjunto>/manifest.json --rate 1000
```

**Catálogo.** Edición `a459dbb8…`, objetivos `fe86a923…`. 18.698.976 filas de 5.008 activos en 18 grupos de filas, 257 MB y huella `6edfddf9…`. La construcción tardó 3 min 19 s con 1,73 GiB de memoria residente máxima. La verificación independiente releyó y volvió a resumir los 59 GB de muestras y etiquetas en 36 min 37 s con 0,62 GiB, limitada por el disco. Ninguna fila llega a 2024.

**Subconjuntos.** Las tres definiciones de `configs/targets/edition-subsets/` se materializaron y verificaron con el código final. La edición sin ajustar de la liquidez es `1ac37278…`.

| Definición | Filas | Activos | Identificadores | Materializar | Verificar |
| --- | ---: | ---: | --- | ---: | ---: |
| `usable-2000-2023` | 18.382.961 | 5.005 | `fbab2865…`, 27 MB | 1,79 GiB | 2,18 GiB |
| `liquid-usable-2000-2023` | 15.724.204 | 4.366 | `42e65a1d…`, 20 MB | 1,69 GiB | 2,08 GiB |
| `liquid-usable-half-2000-2023` | 7.862.112 | 4.366 | `b88e98cb…`, 13 MB | 1,56 GiB | 1,75 GiB |

La memoria es la residente máxima de cada proceso. Las mismas definiciones con la primera versión del código dieron identificadores, universos y recuentos idénticos byte a byte, con 2,30, 2,93 y 3,63 GiB al materializar y 2,76, 3,36 y 3,82 GiB al verificar. El recibo guarda también los tiempos de proceso de las dos ejecuciones, pero no se pueden comparar entre sí: la final se hizo con el equipo muy cargado por otras sesiones (carga media de 19 a 29 al empezar y al terminar), y la primera con una carga menor que no se registró.

**Dimensionado.** Con la campaña A v2 de develop y el mismo caudal uniforme de 1.000 muestras-época por segundo, el subconjunto líquido ajusta 243.916.412 filas-ventana de `train` frente a 285.816.010 del diseño completo, y las horas neuronales quedan en 0,8537 de las del diseño completo. La mitad líquida ajusta 121.958.476, un 0,4269. Las filas son las mismas que antes de integrar #525. Los cocientes cambian en la quinta cifra porque esa campaña declara las correcciones B6 sin ajuste, solo con predicción. La parada conjunta de #530 no los cambia.

## Antes y después

La primera versión de la selección guardaba por fila el activo, el año, la partición, el mercado y la huella en 64 bits, agrupaba los pares de activo y año con `np.unique` sobre filas y convertía la partición en texto para formar los estratos. La versión actual guarda códigos compactos, agrupa con claves enteras de rango pequeño marcando y contando, y ordena los grupos en el tipo sin signo más pequeño. El dimensionado rehacía los cortes de cada ventana para cada activo y ahora los prepara una vez por ventana. Para medirlo sin depender de la carga, `before_and_after` del recibo intercala en procesos separados la versión anterior, reconstruida en una copia aparte, y la actual:

| Medida | Anterior | Actual |
| --- | ---: | ---: |
| Selección de `usable-2000-2023` | 27,3 y 21,5 s, 2,30 GiB | 17,9 y 20,3 s, 1,78 GiB |
| Selección de `liquid-usable-half-2000-2023` | 228,3 y 214,9 s, 3,62 GiB | 97,2 y 118,2 s, 1,56 GiB |
| Recuento de CN del dimensionado con la mitad líquida | 151,1 y 183,1 s, 1,10 GiB | 52,2 y 100,2 s, 1,10 GiB |

Las dos versiones escriben los mismos identificadores y los mismos recuentos. La carga media del equipo estuvo entre 15 y 28 durante la comparación, así que la dispersión es grande y las cifras solo ordenan el coste. En la mitad líquida, un perfil con cProfile atribuía antes 32,5 s de 63,6 al universo (19 s de la asignación de estratos y el resto de `np.unique` sobre filas) y 18,6 s a los estratos. Después, de 27,6 s, quedaron 19,1 s en el universo, casi todo la asignación, y los estratos bajaron de 0,5 s. La asignación pertenece a `evaluation.liquidity_strata`, cuyo código forma parte de las huellas de los informes de evaluación, y no se ha tocado. Lee el Parquet de precios de cada uno de los 5.005 activos y es ahora la mayor parte del coste. Ordenar con Polars fue más rápido con miles de grupos y más lento con decenas de estratos, así que se mantiene `np.lexsort`. No se midió la energía, por falta de instrumento.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 38 mutaciones aplicadas de una en una sobre una copia aislada de `src`, `tests` y `configs`, con las pruebas importando la copia. Las 38 hicieron fallar las pruebas del catálogo, de los subconjuntos, de la liquidez, de las vistas con subconjunto o de las vistas por ventana, con la copia sin mutar en verde (79 pruebas y una omitida por el bloqueo de aprendizaje):

- En el corte de liquidez: el año siguiente o el anterior, `keep` invertido o ignorado, el máximo por activo antes del universo, recuentos duplicados, identidad de la edición, sección del asignador, universo publicado y orden de `keep` sin comprobar, pares de activo y año descodificados al revés y posiciones de presencia desplazadas.
- En la selección: semilla o activo fuera de la clave de orden, rango inclusivo, grupos en 8 bits, estratos sin base mixta, redondeo hacia abajo, etiquetas del corte anual perdidas, último año excluido, modalidades con «o», mercados ignorados y pertenencia invertida.
- En el catálogo: límite del test movido, fila de precios que cierra la ventana ignorada, posiciones de etiqueta desplazadas, bit anual invertido, sesión desplazada y comprobaciones de presencia y de enlaces omitidas en la verificación.
- En vistas, observación, campaña y dimensionado: filas fuera del subconjunto conservadas, motivo perdido, cortes de la primera ventana en todas las ventanas de las vistas o del dimensionado, observación sin restringir, huella del subconjunto de la campaña y de las vistas sin comprobar y subconjunto ignorado al contar.

La primera ejecución dejó viva la fila de precios que cierra la ventana. Se añadió una prueba que adelanta un segundo los bloques y exige que `inputs_available_at` sea la disponibilidad de esa fila.

## Pruebas

Una sola sesión de pytest en serie, en CPU y con dos hilos, sobre las pruebas del catálogo, los subconjuntos, la liquidez, las vistas con subconjunto, `temporal_corpus`, las vistas por ventana, las entradas históricas, la unión conjunta, `corpus_inputs`, `campaign_plan`, `masked_campaign`, la campaña A conjunta, la cadena, la parada conjunta, el corpus con máscaras, las entradas temporales, el corpus conjunto, la campaña de referencia y `tests/test_suite_support.py`, sobre la rama integrada con develop: 582 pasan y 119 se omiten porque llegan a un punto de entrada de ajuste protegido por el bloqueo de aprendizaje. Ninguna ejecuta un paso de optimizador.
