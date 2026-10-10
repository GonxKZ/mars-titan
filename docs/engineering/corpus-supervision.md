# Lectura supervisada del corpus

Las etiquetas se guardan por activo. El lector no construye un diccionario con
las etiquetas de todas las empresas ni carga una matriz global de características.
Cada muestra conserva precios, noticias, gráficos, fundamentales y contexto macro.

`prepare_corpus_targets` consume un manifiesto `materialized_corpus` con las
rutas de muestras, sus activos, calendarios y factores de mercado. Comprueba las
huellas de precios y representaciones y exige noticias completas verificadas.
Un factor declarado para Estados Unidos no se acepta como factor chino.

La etiqueta mantiene la referencia OLS en float64, con 252 sesiones y al menos
126 pares. Su objetivo es el retorno apertura-cierre de la siguiente sesión,
menos el intercepto y la exposición al mercado estimados con el pasado. No se
calculan etiquetas posteriores a 2023 para esta campaña.

La preparación utiliza el motor `numpy`, que convierte las fechas una sola vez
y conserva las operaciones de la referencia. La función de
`mars_titan.training.corpus_targets` acepta `backend="reference"` para contrastar
el recorrido anterior. La identidad incluye el motor y ambas implementaciones.
Cambiar esa identidad exige una salida nueva, sin modificar las ediciones ya
entrenadas. La [medición sobre 66 activos](../../reports/resources/residual-preparation.md)
compara tiempo completo, memoria y huellas de los Parquet.

Hay una fila de etiquetas por cada muestra materializada. Las filas no utilizables
conservan un motivo. Omitir una fila y reducir el recuento del manifiesto no basta
para convertirla en una población válida. Una etiqueta que madura en 2023 no se
usa para entrenar con una decisión de 2022. Las decisiones de 2024 en adelante
quedan reservadas sin asignarles un objetivo.

## Lectura y continuación

`CorpusDataset` valida una edición inmutable. Reutiliza las huellas mientras no
cambien tamaño, inode, dispositivo o fechas del archivo. Si cambian, vuelve a
contrastar el contenido. La identidad del manifiesto forma parte del cursor.

El orden de activos y grupos es reproducible a partir de semilla y época. En
entrenamiento se permutan también las filas de cada grupo. No equivale a una
permutación uniforme de todas las filas del corpus. Cada época conserva la
población completa y el último lote parcial.

El cursor indica activo, grupo, posición y número de muestras consumidas. Se
entrega con el lote, separado de cualquier lectura posterior. Reanudar con el
cursor confirmado vuelve al siguiente ejemplo aunque se hubiera adelantado
la lectura de otro lote. El entrenador debe guardarlo después de aplicar la
actualización correspondiente, no al recibir datos por anticipado.

Un archivo de muestras puede contener grupos Parquet sin filas. En la edición
histórica v3, los 25 activos cuyo número de muestras es múltiplo de 128 (el
tamaño de sus grupos) terminan con un grupo vacío. La edición no se reescribe.
Los lectores omiten esos grupos sin renumerar los demás, así que posiciones,
offsets y grupos de origen siguen siendo los físicos. El orden aleatorio permuta
solo los grupos con filas y obtiene la semilla de cada grupo de su rango entre
ellos. En un archivo sin grupos vacíos el rango coincide con el índice, de modo
que el orden, los lotes y los cursores no cambian. En uno con grupos vacíos, el
recorrido y el cursor coinciden con los del mismo archivo sin ellos. Un grupo
con filas sigue validando todas sus dimensiones. Al preparar las vistas, un
activo cuyos grupos no tienen filas se rechaza igual que uno sin grupos.

Los grupos de características se limitan a 64 MiB decodificados. Los precios y
las etiquetas se cargan por activo bajo presupuestos explícitos, de 200.000 y
1.000.000 de filas respectivamente, además del límite de 64 MiB. No son límites
del corpus. El manifiesto admite hasta 8 MiB y las cabeceras Parquet tienen el
mismo límite. Las ventanas de precios se calculan al consumir el lote y no se
duplican en disco.

## Alcance de las ediciones

Una edición `development_snapshot` puede servir para comprobar ejecución y
recuperación mientras siga pendiente la curación. No se presenta como la
comparación completa. El tipo `full_corpus` requiere que el manifiesto declare
completa su población. Esa declaración debe proceder del proceso de curación,
no de cambiar una etiqueta para superar un control.

El lector conserva la separación entre mercados en claves como `US/AAA` y
`CN/AAA`. El cruce opcional toma exclusivamente observaciones de la otra bolsa
ya disponibles en UTC. Conserva unidad, presencia y antigüedad. Un cierre
estadounidense posterior no puede entrar en una decisión china de esa mañana.

La compactación adicional de contextos macro compartidos y la recuperación de
todas las fuentes siguen siendo trabajos separados. Este lector utiliza los
Parquet materializados y no convierte una fuente pendiente en una muestra válida.

La implementación usa el acceso por grupos de [PyArrow](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html).
Las pruebas incluyen 130 activos, un corpus sintético de 104.000 filas,
continuación con prelectura, exclusiones, modificaciones de artefactos y cruces
entre bolsas. Los datos sintéticos comprueban el lector, no la predicción bursátil.
