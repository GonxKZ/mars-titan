# Preparación histórica de 2000 a 2023

El recorrido de la edición `historical_masked_2000_v1` ha terminado para los 5.676 candidatos de la copia auditada. Se han preparado los 5.023 activos con precios y se han identificado 653 sin ellos. La conciliación confirma que se conservan todos los activos y todos los recuentos de precios de la [auditoría anterior](historical-price-windows.md). No se han creado observaciones anteriores a la historia real de cada activo.

Estas salidas son tablas normalizadas y paneles macro. Todavía faltan su codificación conjunta, los objetivos válidos y la verificación de las mismas filas en las ventanas temporales de cada comparación. El aprendizaje continúa bloqueado y la reserva de 2024 permanece cerrada.

La [codificación acotada](historical-encoding-budgets.md) permite pausar por activo y omitir la copia secundaria de gráficos en caché. Conserva los vectores en Parquet y registra los límites de disco, CUDA y lotes antes de completar el recorrido.

La primera edición codificada y verificada reúne 11.436 muestras de dos activos. La edición posterior con lotes de ocho textos tiene 96 activos y 341.883 muestras verificadas. Sus Parquet ocupan 1.188.112.864 bytes y la caché conserva 85.043 vectores de noticias. Los gráficos permanecen en los Parquet sin otra copia en caché. Se han conciliado todas las ventanas válidas de ese prefijo, las máscaras, la disponibilidad y las huellas. Este recuento parcial no acredita los 5.023 activos. El [recibo](../../reports/data/historical-encoding-progress-20261008.json) conserva ambas ediciones y el contraste de tamaños de lote que precede a la ampliación.

## Tablas preparadas

| Mercado | Candidatos | Activos preparados | Sin precios | Filas de precios | Noticias admitidas | Hechos contables admitidos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 4.784 | 4.213 | 571 | 16.090.522 | 2.156.832 | 3.027.615 |
| CN | 892 | 810 | 82 | 2.891.924 | 553.392 | 0 |

Los precios US comienzan el 3 de enero de 2000 y los CN el 4 de enero de 2006. Ambos terminan el 29 de diciembre de 2023. Cada activo conserva su propio comienzo y sus huecos. La ausencia de noticias o cuentas no elimina el activo. Entre los activos US preparados, 1.537 carecen de fuente contable y 90 de fuente de noticias.

Los archivos contables CN contienen 202.767 registros sin publicación acreditada. Esta pasada los identifica y no los convierte en entradas conocidas en el pasado. La incorporación separada de comunicados oficiales revisados sigue pendiente de materialización sobre este padre. Los archivos originales permanecen intactos. Los hechos US repetidos se deduplican según el contrato existente. Se excluyen 9.224 registros cuyo periodo termina después de su presentación, conservando su procedencia y el resto válido de cada empresa.

Se han verificado 25.115 artefactos, con 4.017.049.727 bytes en total, mediante SHA-256 y recuentos Parquet. La lectura de fechas y valores contables utiliza lotes de 512 filas. Comprueba que no entren disponibilidades de 2024, valores contables no finitos ni periodos posteriores a la presentación. El lote decodificado de mayor tamaño durante esa comprobación ocupó 94.056 bytes. Esta cifra corresponde a las columnas verificadas, no al tamaño de una muestra multimodal completa.

El [recibo de preparación](../../reports/data/historical-preparation-20261008.json) identifica el manifiesto, el commit del runtime, la auditoría de precios y las comprobaciones. La pasada anterior interrumpida se conserva con su [recibo propio](../../reports/data/historical-preparation-progress-20261007.json).

El [enriquecimiento posterior](historical-accounting-context.md) añade 287 hechos contables revisados a 49 activos CN. Conserva los 5.676 candidatos, todos los precios y noticias y los 4.974 preparados restantes idénticos al padre. Su recibo separa esta derivación de los recuentos iniciales de la tabla y mantiene pendientes la codificación y los objetivos.

## Paneles macro con ausencias

Cada sesión conserva las 140 posiciones del catálogo. Un valor ausente sigue siendo nulo y conserva su causa. No se reduce la población histórica a la intersección de sesiones completas.

| Mercado | Sesiones | Posiciones macro | Observadas o calculadas | Ausentes | Sesiones con los 140 valores |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | 6.037 | 845.180 | 612.388 | 232.792 | 397 |
| CN | 5.816 | 814.240 | 590.674 | 223.566 | 387 |

Los dos paneles concilian exactamente con sus calendarios y contienen una sola fila por sesión e indicador. Ninguna celda con valor incumple las comprobaciones existentes de disponibilidad, periodo, unidad, procedencia y ajuste histórico. Los valores cero observados se distinguen de las ausencias. La comprobación no vuelve a auditar por sí misma todos los documentos fuente.

El recálculo reutiliza las adquisiciones conservadas y las fórmulas integradas. Primero calcula el catálogo base y las recuperaciones de petróleo, condiciones financieras, versiones STLFSI, GSCPI y comunicados chinos. Después compone las columnas. Los cambios diarios de 21 observaciones usan el contrato `valid_observations` ya contrastado. DFF conserva su definición en días naturales. La composición de STLFSI3 y STLFSI4 respeta el comienzo acreditado de cada versión. Los boletines H.15 de enero de 2000 siguen sin admitirse como valores observados mientras falten sus metadatos históricos de ajuste.

En 2000 hay observaciones de 63 conceptos a lo largo del año, con 14.700 celdas observadas en US y 13.961 en CN. Esto no significa que las 63 estén presentes en cada sesión. El [recibo macro](../../reports/data/historical-macro-panel-20261008.json) conserva el desglose anual de los dos mercados y las causas de ausencia. Las 397 y 387 sesiones completas se usan como diagnóstico de la comparación estricta, sin filtrar la edición histórica.

En el solapamiento de 2022 y 2023 se compararon 70.140 filas US y 67.760 CN con las ediciones anteriores. Coinciden todos los valores, periodos, unidades, causas y hashes de fuentes. En CN coinciden también todas las disponibilidades. En US cambian seis fechas de disponibilidad al principio de 2022. Una reproducción acotada confirma que el calendario anterior, iniciado en 2022, asignaba publicaciones de diciembre de 2021 a su primera decisión. El calendario desde 2000 conserva su disponibilidad anterior. No se modifica la edición previa.

Se contrastaron 214 cálculos de seis diferenciales con una referencia Decimal, seleccionando un caso de periodo común por año disponible y mercado. Se comprobó también que la disponibilidad del resultado fuera el máximo de sus componentes. No se afirma una revisión independiente de todas las fórmulas ni una certificación de la aritmética en coma flotante.

## Ejecución y memoria

La preparación utiliza Parquet por activo y conserva manifiestos separados para recuperar activos ya confirmados. El cálculo macro procesa componentes y meses, con destinos independientes. Los dos paneles finales ocupan 7.143.344 bytes. No se necesita cargar todo el corpus en RAM ni en VRAM para prepararlo o verificarlo.

| Operación | Tiempo de proceso o cálculo | Pico de RAM residente |
| --- | ---: | ---: |
| Preparación completa | 4.004,241 s de proceso | 366,21 MiB |
| Verificación de tablas | 33,56 s de comprobación | 175,53 MiB |
| Componentes macro US | 127,68 s de cálculo | 1.080,46 MiB |
| Componentes macro CN | 107,01 s de cálculo | 1.076,88 MiB |
| Verificación de ambos paneles | 4,60 s de comprobación | 270,50 MiB |

La preparación tuvo una cuota de dos CPU, dos hilos por biblioteca y un límite de 3 GiB para proceso y caché. El servicio alcanzó ese límite y registró 50,4 MiB de intercambio según systemd. El pico de RAM residente no incluye toda la caché contabilizada por el grupo de recursos. La comprobación macro limitó DuckDB a 512 MB. Hubo otras cargas locales, por lo que estos tiempos no son un benchmark aislado ni prueban una aceleración frente a otra implementación. No se ejecutaron modelos ni se utilizó CUDA en estas operaciones.

El recorrido completo ejecutó `mars_titan.data.corpus_preparation` con `original_audited`, ambos mercados, `historical_masked_2000_v1` y el estado de precios auditados. El recálculo usó `recalculate_macro` con `start=2000-01-01`, `history_start=2000-01-01` y `end=2023-12-31`. La composición utilizó `compose_macro_edition` y `compose_stress_history`. `assess_macro_completeness` produjo el diagnóstico separado de completitud. Los recibos conservan versiones, configuraciones, hashes de componentes y tiempos medidos. Los datos voluminosos quedan fuera de Git.

La edición histórica de aprendizaje sigue pendiente de unir las cuentas revisadas, codificar todas las ventanas válidas y comprobar objetivos, máscaras, exclusiones y cortes comunes. Estos resultados no miden mejoras predictivas ni habilitan el entrenamiento.
