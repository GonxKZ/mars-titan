# Preparación histórica con máscaras de ausencia

La política `historical_masked_2000_v1` prepara una edición adicional a partir de los activos con precios. Conserva las ventanas de 64 sesiones aunque falten noticias, fundamentales o contexto macro. La política predeterminada `strict_inputs_v1` mantiene la intersección de entradas anterior. Las dos ediciones usan destinos e identidades distintos.

Esta entrega implementa preparación y codificación de muestras. No calcula objetivos nuevos, adapta lectores de entrenamiento, entrena modelos ni acredita que la edición histórica completa esté lista. La reserva desde 2024 permanece cerrada. El [recuento de ventanas de precios](historical-price-windows.md) es una referencia para conciliar la materialización posterior, no un recuento de muestras supervisadas.

## Entradas, ausencias y fechas

Los precios válidos siguen siendo obligatorios. Se conservan OHLCV, el calendario del mercado y 64 sesiones consecutivas. No se rellenan huecos. Solo se emiten decisiones desde 2000 y anteriores a 2024. El contexto anterior a 2000 puede aportar historia, pero no genera decisiones anteriores al corte.

El preparado identifica fuentes opcionales inexistentes y crea tablas vacías con esquema. Los errores de archivo, huella, calendario, parser o codificador siguen siendo errores. No se transforman en ausencia para confirmar un activo.

Las muestras incorporan estas columnas:

| Columna | Tipo y contenido |
| --- | --- |
| `presence` | Cinco booleanos en el orden precios, noticias, gráficos, fundamentales y macro |
| `missing_reasons` | Causa por bloque ausente, nula si hay alguna observación utilizable |
| `fundamental_missing_reasons` | Causa por concepto contable, con el orden del catálogo |
| `macro_missing_reasons` | Causa por indicador, con el orden del catálogo macro |

Los vectores de noticias ausentes se rellenan con cero y tienen `news_count=0`. Esto cuenta eventos admitidos, no demuestra que no existieran noticias. El gráfico se regenera únicamente desde la ventana pasada de precios y conserva su hash. Un fallo al generarlo o codificarlo impide confirmar la materialización.

Fundamentales y macro mantienen los tres bloques existentes: valores transformados, máscaras observadas y edades transformadas. Un cero real conserva la máscara observada. Un valor ausente tiene relleno cero, máscara falsa y fecha de disponibilidad nula. Los valores sin publicación conocida o publicados después del corte no se admiten. Los NaN e infinitos provocan un error, incluso cuando también existe una causa de ausencia. Las causas declaradas por la fuente se conservan como diagnóstico acotado y no se añaden al vector numérico.

Una fuente macro inexistente necesita un catálogo explícito. El vector conserva esas posiciones con máscaras falsas. Un archivo que existe pero está dañado no se trata como fuente inexistente. Un panel parcial puede completar posiciones ausentes del catálogo, sin inventar valores. El materializador histórico rechaza filtros de sesiones completas y ventanas distintas de 64.

## Identidad y reutilización

La política y `mask_contract` forman parte de la configuración usada para calcular las huellas, junto con los catálogos y las fuentes. El contrato de máscaras es versión 1, fija relleno cero, fechas ausentes nulas y los cortes de 2000 y 2024. Cambiar catálogo, orden, revisión de código o política exige una nueva edición.

El preparado global histórico tiene `schema_version=2`, sus preparados y muestras por activo tienen versión 4 y el corpus materializado tiene versión 3. Los consumidores actuales que solo aceptan formatos anteriores rechazan estos archivos. `training_ready` permanece falso. `cohort_complete` indica que se ha recorrido y conciliado el censo sin errores, no que todas las modalidades estén observadas.

`prepare_cohort` acepta `price_audit_state` para reutilizar los Parquet normalizados. Comprueba la identidad del activo, hash de origen, hash y recuento del derivado, calendario y valores OHLCV. Rechaza enlaces intermedios y archivos o grupos que excedan 64 MiB. Las sesiones se leen por lotes con diccionario y se valida su longitud antes de convertirlas a cadenas Python. Después se decodifica solo el prefijo de valores anterior al corte. Las fechas posteriores pueden leerse como metadatos para delimitar ese prefijo, pero sus valores no se materializan.

La preparación conserva el censo completo. Un activo sin precios queda en `missing_required_prices`. Las ausencias opcionales ya no producen ese estado ni eliminan las ventanas. Los recuentos de exclusión de precios y los de ausencia de entradas se publican por separado.

La CLI de preparación admite `--input-policy historical_masked_2000_v1` y `--price-audit-state`. La CLI de codificación admite la misma política y `--macro-catalog` para declarar el catálogo, también cuando no hay archivo macro. Se reutilizan los codificadores congelados y las cachés existentes. Las comprobaciones técnicas utilizan codificadores de prueba explícitos, sin cargar pesos ni producir evidencia financiera.

Las huellas de los artefactos históricos no se reescriben. Una ejecución con código nuevo obtiene una identidad nueva aunque produzca exactamente los mismos vectores estrictos. La prueba de paridad compara esquema, filas y bytes del Parquet estricto anterior y posterior al cambio.

## Dependencias pendientes

La supervisión mantiene su objetivo residual original y su historia de 252 sesiones con un mínimo de 126 pares. Esta entrega no lo modifica. La falta de objetivo válido seguirá separada de la ausencia de noticias, cuentas o macro.

Faltan la materialización y conciliación del corpus real, la supervisión de las filas nuevas y los contratos de lectura por fase. Los consumidores tabulares, neuronales, RL y el candidato necesitan una adaptación explícita a las máscaras y la misma lista de filas. No deben convertir presencias falsas en verdaderas ni filtrar poblaciones diferentes por modelo.
