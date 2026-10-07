# Ocupación temporal de la preparación US y CN

Se midió una preparación nueva de entrenamiento y validación de la ventana
`fold-009` de la edición conjunta. Recorrió 321.610 y 43.185 filas,
respectivamente, sin ampliar ni repetir la población para provocar escrituras
temporales. La fuente sigue siendo una `development_snapshot`. La medición no
es un entrenamiento ni acredita cobertura completa del universo chino.

El [recibo saneado](../../reports/resources/corpus-temporary-storage-20261007.json)
identifica fuente, código, configuración y comprobaciones. Los perfiles originales
permanecen fuera del repositorio porque contienen rutas locales.

## Medidas distintas

| Medida | Entrenamiento | Validación |
| --- | ---: | ---: |
| Pico nativo de temporales de DuckDB, bytes | 0 | 0 |
| Pico nativo de buffers de DuckDB, bytes | 5.224.854.560 | 764.269.236 |
| Archivo de entrada temporal, bytes lógicos | 2.060.963.783 | 262.141.675 |
| Archivo ordenado, bytes lógicos | 1.392.366.107 | 187.659.929 |
| Tiempo de `COPY` y ordenación, segundos | 19,66 | 2,21 |

El motor no necesitó desbordar sus buffers a disco en esta carga. Eso no elimina
los archivos intermedios de la preparación. El máximo observado de estos archivos
fue **3.453.329.890 bytes lógicos** y **3.453.341.696 bytes asignados**. Se alcanzó
con `input.parquet` y `ordered.parquet` coexistiendo tras ordenar entrenamiento.
El primero se elimina al terminar la partición y el segundo se conserva mediante
el enlace de publicación del productor.

La preparación completa tardó 110,83 segundos y alcanzó un RSS de
8.308.379.648 bytes, unos 7,74 GiB. El contador de buffers de DuckDB no representa
todo ese RSS. La admisión inicial del lector alcanzó 673.411.072 bytes y su caché
se limitó a 64 MiB. El sondeo observó al menos 8.646.541.312 bytes disponibles en
el equipo durante la preparación, por encima del margen de parada de 2 GiB.

El intervalo completo del servicio, incluido el arranque del proceso, fue de
112,91 segundos. Systemd registró 122,51 segundos de CPU, un pico de memoria del
cgroup de 8.123.224.064 bytes y cero intercambio. El RSS del proceso y la memoria
cargada al cgroup son contadores diferentes y se conservan por separado.

Los contadores de `/proc/self/io` registraron 12.528.456.569 bytes devueltos por
lecturas (`rchar`) y 3.903.331.069 bytes entregados a escrituras (`wchar`). Los
contadores `read_bytes` y `write_bytes` fueron 1.442.848.768 y 3.903.385.600 bytes.
Son tráfico acumulado del proceso y están afectados por la caché, no ocupación
simultánea de disco. En los perfiles de ambos `COPY`, el contador nativo
`TOTAL_BYTES_WRITTEN` fue cero aunque se escribieron los Parquet. Por eso no se
utiliza como medida de todas las escrituras de esta ruta.

## Procedimiento reproducible

La medición reutiliza
[`prepare_causal_corpus`](../../src/mars_titan/environments/corpus_source.py),
con lotes de 256, una salida nueva y sin cambiar la fuente supervisada. El arnés
de medida envuelve la conexión que abre `_partition`. Solo cambia su número de
hilos de cuatro a dos y construye `CorpusDataset` con `cache_bytes=64*1024**2`.
Conserva `memory_limit='8GiB'`, `max_temp_directory_size='32GiB'`, orden por
`prediction_at, asset_id`, compresión Zstandard y el tamaño de grupo existente.

DuckDB 1.5.5 ofrece perfiles JSON y el contador `SYSTEM_PEAK_TEMP_DIR_SIZE`, en
bytes. La cobertura `ALL` permite recoger `COPY`, además de las consultas
`SELECT`. Dentro de la conexión, la instrumentación usada es:

```python
connection.execute("SET profiling_coverage = 'ALL'")
connection.execute("SET enable_profiling = 'no_output'")
started = time.perf_counter()
connection.execute(query, parameters)
elapsed = time.perf_counter() - started
profile = json.loads(connection.get_profiling_information("json"))
connection.disable_profiling()
```

`query` y `parameters` son los del `COPY` original del productor. El perfil se
captura inmediatamente después, antes de que otra consulta lo sustituya. No se
ejecuta otra ordenación mediante `EXPLAIN ANALYZE`. Las opciones y unidades se
describen en la documentación oficial de
[profiling](https://duckdb.org/docs/current/dev/profiling) y
[métricas](https://duckdb.org/docs/current/dev/metrics).

Para repetir el resto de la medición:

1. Usar el commit y las versiones del recibo, con el entorno ya preparado,
   `uv run --no-sync`, `PYTHONPATH=src` y `CUDA_VISIBLE_DEVICES=-1`. Fijar un hilo
   para OpenMP, OpenBLAS y MKL, dos para DuckDB y lectura Arrow sin paralelismo
   interno de lotes. La preparación no carga pesos ni ejecuta inferencia.
2. Comprobar al inicio al menos 13 GiB en `MemAvailable`, antes de importar las
   bibliotecas que contarán dentro del proceso. Limitar el servicio a dos CPU,
   11 GiB de memoria y sin intercambio. Tras abrir el lector, exigir un pico RSS
   menor de 1,5 GiB para conservar margen para lotes y decodificación fuera de
   DuckDB. No reducir los 8 GiB del motor para fabricar presión de disco.
3. Registrar cada 250 ms el tamaño lógico (`st_size`) y asignado
   (`st_blocks*512`) de los directorios de trabajo de esa preparación. Deduplicar
   por dispositivo e inode para no sumar dos veces enlaces al mismo archivo.
   Conservar por separado los archivos de `spill`. Repetir la observación antes
   y después de cada `COPY`. Esta ejecución produjo 446 observaciones.
4. Mantener un margen de 2 GiB disponibles. Si cae, solicitar `StopRequest` e
   interrumpir únicamente la conexión DuckDB propia. Convertir ese corte en una
   interrupción recuperable por partición. No señalizar aplicaciones ajenas.
   En esta ejecución no fue necesario pausar.
5. Medir el RSS mediante `getrusage` y restar los contadores de E/S del proceso
   antes y después de preparar. Guardar perfiles por partición y sus huellas
   antes de que se eliminen los archivos de trabajo. Separar la comprobación
   posterior de paridad del tiempo y tráfico de la preparación.
6. Invocar la misma preparación con `resume=True` y comprobar huellas, tamaño,
   inode y fecha de modificación de las particiones confirmadas. Contrastar
   todas las filas ordenadas con el lector supervisado por clave y por bytes de
   modalidades, objetivo y marcas temporales.

La recuperación tardó 4,55 segundos, no escribió bytes según los contadores del
proceso y conservó ambos Parquet sin reescritura. La comprobación independiente
posterior verificó las 364.795 claves, sin duplicados ni omisiones. Sus SHA-256
por fila coinciden para las cinco entradas numéricas, incluido el contexto macro,
la etiqueta, las fechas, el activo y la cohorte. Esa comprobación tardó 57,91
segundos y no forma parte de los 110,83 segundos de preparación.

## Alcance de los picos

El pico de temporales nativo procede del contador del motor. Los tamaños de la
carpeta de trabajo son máximos observados mediante sondeo y barreras, no una
medición continua de su pico exacto. Sus 446 observaciones y las comprobaciones
tras `COPY` no encontraron archivos de spill, de acuerdo con el contador nativo.
No se suman máximos de fases diferentes como si hubieran coexistido.

Es una ejecución con la caché del sistema sin vaciar y otras cargas del equipo
sin detener. No estima variabilidad, aceleración ni consumo energético. Conserva
las cuentas de calibración y evaluación del padre como metadatos, pero solo
materializa entrenamiento y validación. El test final de 2024 permanece cerrado.
