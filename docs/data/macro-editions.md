# Componer una edición macro sin alterar sus fuentes

`mars_titan.data.macro_edition` reúne el panel anterior y las columnas recuperadas en una edición distinta. Cada sustitución declara su archivo Parquet y sus identificadores. Un indicador solo puede pertenecer a una sustitución. El proceso exige exactamente una fila por indicador y sesión, conserva los ceros observados y deja intactas las ausencias explicadas por las fuentes.

La composición recorre cada fuente en orden temporal y reúne un mes cada vez. Arrow ordena las filas por decisión e indicador y escribe Parquet con Zstandard. Los grupos de entrada y el bloque mensual tienen presupuestos de memoria, y el pie de metadatos se comprueba antes de abrir el lector. No se carga el panel entero en una tabla ni se descargan datos durante la composición.

`uv run python -m mars_titan.data.macro_edition --help` describe la interfaz. `--replacements` recibe una lista JSON cuyos elementos contienen `path` e `indicator_ids`. Las rutas de ejecución permanecen en la configuración local. El recibo de salida registra las huellas de todos los archivos, el catálogo y las columnas sustituidas. Las fuentes se vuelven a comprobar antes de publicar el directorio. Un destino existente no se sobrescribe.

La composición no implica que la población esté admitida. `report.json` indica `admission_required=true`. Después se aplica la [puerta de cobertura](macro-admission.md), que comprueba valores finitos, disponibilidad, unidades, procedencia y presencia de todos los indicadores. La selección de empresas y modalidades se comprueba por separado.

## Comprobación de la edición con 140 indicadores

La ejecución local anterior a 2024 combina quince columnas recuperadas con las demás columnas de la edición anterior. Produce 528.360 filas para 3.774 sesiones. Una consulta independiente en DuckDB ha comparado las nueve columnas mediante diferencias de multiconjuntos en ambos sentidos, sin discrepancias.

La admisión encuentra 257 sesiones con los 140 indicadores válidos, entre el 11 de noviembre de 2022 y el 26 de diciembre de 2023. Otras 3.517 sesiones quedan excluidas y no hay decisiones con indicadores duplicados. Las primeras fechas no pueden completarse trasladando retrospectivamente índices que aún no se publicaban. Las 257 sesiones tampoco garantizan que cada empresa tenga muestras y etiquetas admisibles en todas ellas.

Tras un primer recorrido, tres ejecuciones con caché local tardaron entre 1,23 y 1,28 segundos dentro de la composición y entre 1,65 y 1,86 segundos contando el proceso. El máximo observado fue 173,88 MiB de RSS. El bloque mensual mayor ocupó 581.774 bytes. Estas medidas incluyen comprobaciones locales concurrentes y no representan una comparación de aceleración. Las condiciones, el perfil, las pruebas y sus límites se registran en [macro-edition-quality.json](../../reports/resources/macro-edition-quality.json).
