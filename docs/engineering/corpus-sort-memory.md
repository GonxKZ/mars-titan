# Memoria de la ordenación del corpus completo

La preparación de los ajustes predictivos ordena las filas por decisión y activo, conservando conjuntamente precios, noticias, gráficos, fundamentales, contexto macro y etiqueta. La partición de entrenamiento tiene 1.548.307 filas, con 1681 valores float32 por fila. Solo esos valores ocupan 9,70 GiB sin comprimir.

El límite anterior de 512 MiB de DuckDB falló durante `COPY ... ORDER BY`. El primer contraste con 4 GiB también falló y alcanzó 6.050.693.120 bytes de RSS. La configuración de 8 GiB completó entrenamiento y validación en 415,48 segundos, con un pico de proceso de 11.587.465.216 bytes (10,79 GiB). Se conservaron cuatro hilos, el orden explícito y un máximo de 32 GiB de temporales de DuckDB. El servicio permaneció limitado a 12 GiB, incluyendo su caché de archivos.

La salida confirma 1.548.307 filas de entrenamiento y 261.069 de validación. Sus archivos Parquet ocupan 6.307.465.656 y 1.134.561.416 bytes. Las claves, fechas de disponibilidad, maduración de etiquetas, población y dimensiones se comprueban antes de publicar cada partición. Las predicciones padre se alinean después por clave y objetivo, comprobando igualdad exacta. El test final permanece cerrado.

El [informe de ejecución](../../reports/resources/corpus-sort-memory.json) conserva los dos fallos anteriores, la configuración, versiones, huellas y muestras de recursos. El tiempo hasta un fallo no se presenta como referencia de rendimiento de una ejecución terminada. El máximo de RSS incluye el proceso de preparación, y los contadores de I/O incluyen la admisión de sus padres.

El [límite de memoria de DuckDB](https://duckdb.org/docs/current/configuration/pragmas#memory-limit) corresponde al gestor de buffers, no a toda la memoria del proceso. Por eso se comprueban también RSS y memoria del servicio. El resultado de 8 GiB es evidencia para esta carga, no una cota universal para otros universos o dimensiones.

Las ediciones fallidas se conservan. La preparación corregida y sus ajustes utilizan una edición nueva, que admite los mismos padres confirmados por su población y sus artefactos. No se modifican las fuentes ni los resultados neuronales y tabulares terminados.
