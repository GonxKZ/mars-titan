# Contextos macro desde el archivo H.15

`macro_h15_contexts.prepare_h15_contexts` crea un panel adicional a partir de una edición documental H.15 ya confirmada. Reutiliza `calculate_macro`, sus fórmulas y el catálogo. El preparado documental se verifica de nuevo frente a HTML, PDF, revisión, recibos, configuración y Parquet antes de utilizar sus observaciones.

Las filas documentales de US y CN describen los mismos hechos. El adaptador conserva un evento por indicador, periodo y publicación, con su decimal, hash PDF, hash de revisión y ambas disponibilidades. Dos boletines aportan sesenta eventos, aunque el preparado tenga ciento veinte filas. Las publicaciones posteriores del mismo periodo conservan una versión distinta. Dos valores distintos atribuidos a la misma publicación no se sustituyen entre sí.

La fecha del periodo no fecha la disponibilidad. Cada publicación se lleva al final del día en Nueva York y a la siguiente sesión del mercado receptor. Se comprueba que el resultado coincide con el preparado documental. Las decisiones emitidas comienzan en 2000 y terminan antes de 2024.

## Metadatos y alcance del cálculo

El catálogo conserva las seis series originales. DFF mantiene frecuencia D7 y sus retardos se expresan en días naturales. Las series Treasury mantienen frecuencia D. La regla opcional `valid_observations` afecta únicamente a los retardos de observaciones diarias. Ninguna regla añade precios de tipos, fines de semana ni antecedentes ausentes.

El parámetro opcional `metadata_manifest` identifica un ZIP ALFRED por cada una de las seis series. Su JSON tiene `schema_version=1` y un objeto `archives`, cuyas claves son los identificadores del catálogo. Cada referencia contiene `path`, `sha256`, `series_id` y `url`. La ruta es relativa al manifiesto, no puede escapar de su directorio ni atravesar enlaces. El README del ZIP debe confirmar la misma serie y URL oficial. Se utilizan únicamente sus metadatos, no se incorporan sus observaciones.

Se reutilizan los intervalos de unidad y ajuste estacional del README. Ambos deben cubrir la fecha de publicación de H.15. `Percent` y `Not Seasonally Adjusted` son compatibles con el contrato del catálogo. Si falta un intervalo, el evento conserva su cifra documental, pero entrega al motor `value=None`, ajuste nulo y `missing_historical_metadata`. Los intervalos contradictorios conservan `ambiguous_historical_metadata`. Una definición incompatible impide confirmar el panel.

Los metadatos locales revisados de las seis series empiezan el 28 de junio de 2005. Los boletines de enero de 2000 acreditan porcentajes anuales y los vencimientos seleccionados, pero no documentan explícitamente NSA. Por eso esas cifras no se admiten en el panel de enero de 2000 mediante una descripción posterior. Los cuatro diferenciales Treasury pueden contrastarse como aritmética documental separada. Ese contraste no acredita el ajuste histórico ni convierte el resultado en una entrada admitida.

Con entradas y metadatos suficientes, el motor calcula las pendientes 10y−2y, 10y−3m, 30y−10y y 5y−2y del mismo periodo. Los cambios a veintiún días o veintiuna observaciones quedan ausentes si no hay antecedente. Una entrada de 140 conceptos produce las 140 posiciones por decisión, incluidas las ausencias de las demás fuentes.

La unión de observaciones ALFRED requiere otra entrega con reglas explícitas para los solapes y sus versiones. Esta API rechaza `alfred_source`. Un ZIP empleado para acreditar metadatos no autoriza esa unión ni anticipa su política de precedencia.

## Salida y recuperación

La función recibe el manifiesto de fuentes H.15, el directorio documental confirmado, el catálogo y un destino nuevo, junto a `start`, `end`, mercados y metadatos opcionales. La CLI es `python -m mars_titan.data.macro_h15_contexts`, con `--manifest`, `--documents`, `--catalog`, `--output`, `--start`, `--end`, `--metadata-manifest` opcional y `--market` repetible.

La edición contiene `events.json`, un `macro-US.parquet` o `macro-CN.parquet` por mercado, `configuration.json` y `report.json`. La configuración identifica fuentes, código, calendario y política de retardos. El recibo concilia filas, observaciones y causas de ausencia. Mantiene `admission_required=True`, `training_ready=False` y el alcance de archivo retrospectivo fechado. No afirma una captura contemporánea de las publicaciones.

Se permiten hasta un millón de celdas entre todos los mercados, 64 MiB por artefacto y ocho MiB por ZIP de metadatos y su contenido descomprimido. El productor documental conserva sus propios límites. La publicación utiliza un directorio temporal y renombrado sin reemplazo. La recuperación vuelve a verificar las fuentes y recalcula el panel desde los eventos. Compara todas las columnas, sus tipos y su orden mediante lotes de 512 filas. Los textos se mantienen como diccionarios hasta comprobar sus longitudes y presupuesto de expansión. Los artefactos alterados se rechazan y los correctos no se reescriben.

Esta salida no modifica paneles anteriores ni configura una campaña. La preparación de la edición histórica completa y su admisión siguen pendientes.

La comprobación de los dos boletines de enero de 2000 conserva sesenta cifras y sesenta eventos. Produce 1.960 celdas por mercado en catorce decisiones, con los 140 conceptos y ninguna observación admitida en ese intervalo. La falta de metadatos queda explícita. Los cuarenta diferenciales documentales contrastados con Decimal tienen un error absoluto máximo de `6.4e-16`, por debajo de la tolerancia declarada de `1e-14`. El [recibo de la edición](../../reports/data/h15-context-edition-20261007.json) recoge fuentes, medidas, pruebas y límites.

La revisión de integridad reprodujo dos fallos: se podía recuperar un panel modificado si se actualizaba también su hash en el recibo, y un cambio del Parquet documental después de su verificación podía convertirse en la nueva referencia. La recuperación corregida compara el resultado recalculado y el adaptador conserva las huellas del recibo documental ya verificado. La [comprobación posterior](../../reports/data/h15-context-integrity-20261007.json) registra ambos rechazos y una edición nueva cuyos eventos y paneles conservan los bytes anteriores.

La comparación canónica de JSON conserva la diferencia entre booleanos y números, y entre `1` y `1.0`. Se aplica a las identidades, los recibos, los recuentos y los eventos. La [verificación de tipos](../../reports/data/h15-json-integrity-20261008.json) comprueba que esas sustituciones se rechazan y que los archivos legítimos se recuperan sin reescritura.
