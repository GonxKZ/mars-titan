# Cohortes causales desde Parquet

`environments.corpus_source` convierte una edición supervisada en dos archivos
ordenados, uno de entrenamiento y otro de validación. Cada archivo conserva
las claves, etiquetas, maduración y las cinco entradas del modelo. La
ordenación usa el instante UTC y el identificador del activo. Una cohorte
contiene todos los activos admitidos para ese instante.

La preparación no establece un límite de empresas ni toma una submuestra.
Comprueba el número completo de filas de cada partición y la unicidad de
activo e instante. La rejilla predictiva se ajusta solo con las etiquetas de
entrenamiento. La reserva de 2024 en adelante no se exporta.

## Contrato temporal y procedencia

El lector supervisado añade `target_available_at` y `input_available_at` a
cada lote. La primera procede de la etiqueta. La segunda es el máximo de
las disponibilidades declaradas por las modalidades y macro. No se utiliza
el periodo contable como sustituto de la publicación.

Se admiten el formato que guarda cinco fechas dentro de `input_availability`
y el formato histórico que guarda cuatro y conserva `macro_available_at`
como columna separada. La ausencia de esa evidencia queda como `NaT` y
bloquea la exportación causal. Una fecha nula de macro no hereda la fecha
de otra modalidad. El codificador exige al menos un indicador observado y
una fecha acreditada para cada valor macro presente.

La cohorte y su política editorial permanecen en el manifiesto. Cada fila
Parquet conserva además `cohort_id`, que se contrasta durante la lectura.
La cohorte original auditada no se presenta como una verificación externa
de noticias completas. Las ediciones históricas sin etiqueta de cohorte
conservan esa ausencia, junto con la huella de su manifiesto original.

## Ordenación y memoria

La entrada se recorre con `CorpusDataset`, en lotes de 256 filas por defecto.
PyArrow escribe una partición temporal y DuckDB la ordena con cuatro hilos,
un presupuesto del gestor de memoria de 512 MiB y hasta 32 GiB para derrame
temporal en disco. Esos límites no incluyen todas las asignaciones del
proceso Python ni sustituyen su límite externo de RAM.

El [mínimo de tamaño de grupo de DuckDB](https://duckdb.org/docs/current/data/parquet/tips)
es 2.048 filas. Con vectores muy anchos, ese grupo puede superar el presupuesto
del lector. En ese caso se vuelve a agrupar con PyArrow, conservando el orden.
La decisión usa los bytes por fila y los metadatos físicos, y comprueba los
grupos resultantes antes de confirmar la partición. No se reducen dimensiones
ni se eliminan muestras para hacerla caber.

`ParquetCohortSource` mantiene como máximo dos grupos en caché, con un límite
conjunto de 64 MiB por defecto. Lee las porciones contiguas de una cohorte,
comprueba su fecha y sus formas y devuelve un bloque independiente. Las
copias del bloque devuelto y del entorno son adicionales a la caché. No es
una afirmación de que el proceso completo ocupe 64 MiB.

## Publicación y recuperación

Los archivos se nombran por su contenido y se sincronizan antes de publicar
los recibos. `progress.json` confirma cada partición terminada. `--resume`
verifica las huellas y reutiliza las particiones confirmadas. Si se interrumpe
la publicación del manifiesto final, la recuperación puede verificar los
archivos y terminar esa publicación.

Una preparación completada no se reescribe al recuperarla. Los cambios de
datos, código, versiones o lote se rechazan. Las interrupciones dentro de
una partición obligan a repetir esa partición, no todo lo ya confirmado.
La memoria del lector comprueba también la identidad física del archivo
antes y después de las lecturas, incluidas las comprobaciones de huellas.

El módulo acepta `--manifest`, `--output`, `--batch-size` y `--resume`.
La preparación requiere el extra `data`. El entorno requiere además
`reinforcement`. Las dependencias se gestionan con uv y no se sincronizan
mientras un proceso científico utiliza ese mismo entorno.

## Evidencia y límites

La [comprobación de paridad](../../reports/resources/parquet-cohorts-real-audit.json)
recorre la edición verificada anterior de 66 activos. Conserva sus 245
muestras de entrenamiento y 225 de validación. No hay diferencia numérica
en las cinco entradas y coinciden las etiquetas y sus fechas de maduración.
Esta edición sigue identificada como `development_snapshot` incompleta.
No representa la campaña amplia que continúa materializándose.

Las pruebas adicionales cubren una edición de procedencia explícita, grupos
Parquet atravesados por cohortes, presupuesto reducido, archivos alterados
y recuperación durante la publicación final. El
[comparador directo](../../reports/analysis/check_causal_corpus.py) tiene su
propio presupuesto de 64 MiB y falla si la edición supera ese límite.
No se utiliza para recortar una edición científica.

Este adaptador deja disponible el recorrido cronológico. No acredita
entrenamientos de refuerzo, PPO financiero ni la arquitectura MARS-TITAN.

La [evidencia de pruebas](../../reports/resources/parquet-cohorts-quality.json)
registra la batería completa, cobertura, complejidad, CRAP y cuatro mutaciones
dirigidas. Las tres incidencias de revisión tienen regresiones específicas.
