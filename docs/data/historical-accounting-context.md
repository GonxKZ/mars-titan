# Contexto contable común de la edición histórica

La política `historical_disjoint_accounting_v1` utiliza los 26 conceptos de la representación conjunta existente. Conserva 15 canales USD y ratios, ocho CAD y tres CNY. Cada muestra tiene 78 posiciones: valores transformados, máscaras observadas y edades. El orden de los conceptos se conserva en `accounting_catalog.JOINT_CONCEPTS` y forma parte de la identidad de la edición.

US calcula sus canales USD y CAD por separado. Los ratios de los siete canales compartidos se obtienen únicamente de cuentas USD. Una publicación CAD posterior no sustituye un ratio USD por una ausencia ni se promedia con una cuenta USD. CN conserva los conceptos revisados de activos, pasivos y patrimonio con minoritarios en CNY, con norma CAS y perímetro consolidado. Los demás conceptos originales permanecen en las tablas de hechos y quedan fuera de este catálogo de representación.

La proyección usa `project_numeric_context` y conserva valores, máscaras y edades por identificador exacto. No convierte monedas ni renombra conceptos para equipararlos. Un cero observado tiene máscara verdadera. Los canales añadidos al espacio común tienen valor y edad cero, máscara falsa y causa `outside_source_accounting_catalog`.

## Preparación y codificación

`historical_accounting.prepare_historical_accounting` recibe el manifiesto de una preparación histórica completa, una unión CN revisada y un destino nuevo. Rechaza un padre en ejecución, pausado o con errores antes de copiar datos. Recorre todos sus candidatos, incluidos los que no tienen precios. Copia los preparados sin enriquecimiento y deriva los CN identificados por el catálogo revisado.

`reviewed_cn_catalog` obtiene las referencias contables de los recibos de la unión CN. Contrasta configuración, manifiestos y censo de emisores, sin seleccionar filas desde sus muestras o etiquetas anteriores. `derive_chinese_preparation`, con `input_policy="historical_masked_2000_v1"`, vuelve a comprobar los hechos mediante los importadores existentes. Exige un origen contable vacío, identidad del activo, hashes de origen, CNY, CAS, fecha de publicación, siguiente sesión del calendario CN y correspondencia del decimal con su valor almacenado. Las publicaciones y revisiones conservan su propia disponibilidad. Un duplicado o un contexto incompatible provoca error.

El preparado enriquecido mantiene el esquema histórico 4 por activo y 2 para el censo. Conserva las ausencias originales y el detalle de su auditoría contable. La publicación de cuentas revisadas no acredita los originales CN que carecen de unidad o publicación conocida.

La codificación se solicita con `encode_corpus(..., input_policy="historical_masked_2000_v1", accounting_policy="historical_disjoint_accounting_v1")`. La CLI admite `--accounting-policy historical_disjoint_accounting_v1`. Un preparado enriquecido exige esa opción para impedir que se omitan sus canales mediante la configuración contable anterior.

La proyección se aplica por lotes después de formar cada contexto causal. Conserva las mismas sesiones, índices de precios, presencias de modalidad y demás columnas. No usa `complete-decisions` ni elimina filas porque falten cuentas, noticias o macro. La política histórica sigue requiriendo 64 sesiones consecutivas y mantiene cerrado 2024. Las identidades de representación US y CN comparten el catálogo de salida y el código de proyección, mientras que cada activo conserva su moneda y catálogo de origen.

## Recuperación y límites

La preparación registra configuración, procedencia, código y calendarios antes de recorrer el censo. La pausa operativa se controla con `stop_after_assets`. La CLI de preparación acepta `--prepared`, `--reviewed-cn`, `--output` y `--stop-after-assets`. Los activos confirmados se verifican y reutilizan, sin reescribirlos. Un destino con otra identidad o un archivo alterado se rechaza. Los preparados y las ediciones anteriores permanecen separados.

Se admiten hasta 10.000 candidatos y se mantienen los límites del importador de 16 ediciones, 100.000 hechos y 64 MiB por historia contable. La proyección conserva su límite conjunto de entrada y salida de 64 MiB y procesa bloques de hasta 1 MiB. La codificación conserva los presupuestos existentes por activo, noticias y caché. Las copias del preparado se realizan por archivo, con huellas verificadas.

La inspección de recibos y esquemas de la unión CN identifica 49 emisores, 51 ediciones contables y 287 filas declaradas. No se han decodificado sus hechos ni ejecutado el enriquecimiento real. Los recibos de esos emisores son compatibles con las fuentes declaradas por el preparado histórico completado. Las pruebas funcionales utilizan fuentes sintéticas y codificadores de prueba, sin objetivos ni modelos.

Quedan pendientes la derivación real, la codificación autorizada y su conciliación con las ventanas de precios. La proyección contable no acredita objetivos residuales, cobertura macro completa ni aptitud para entrenar. `training_ready` permanece falso y las familias de modelos deberán consumir el mismo manifiesto y sus mismas filas.
