# Preparación histórica de 2000 a 2023

El recorrido de la edición `historical_masked_2000_v1` ha terminado para los 5.676 candidatos de la copia auditada. Se han preparado los 5.023 activos con precios y se han identificado 653 sin ellos. La conciliación confirma que se conservan todos los activos y todos los recuentos de precios de la [auditoría anterior](historical-price-windows.md). No se han creado observaciones anteriores a la historia real de cada activo.

Estas salidas son tablas normalizadas y paneles macro. Su codificación conjunta y los objetivos válidos se completaron y verificaron el 9 de octubre, como recoge la [sección siguiente](#edición-v3-completa-y-objetivos-residuales). Falta conciliar las mismas filas en las ventanas temporales de cada comparación, que se están preparando. El aprendizaje continúa bloqueado y la reserva de 2024 permanece cerrada.

La [codificación acotada](historical-encoding-budgets.md) permite pausar por activo y omitir la copia secundaria de gráficos en caché. Conserva los vectores en Parquet y registra los límites de disco, CUDA y lotes antes de completar el recorrido.

La primera edición codificada y verificada reúne 11.436 muestras de dos activos. La edición posterior con lotes de ocho textos conserva 480 activos y 1.673.363 muestras verificadas. Sus Parquet ocupan 5.704.577.747 bytes y la caché conserva 246.211 vectores de noticias. Los gráficos permanecen en los Parquet sin otra copia en caché. Se han conciliado todas las ventanas válidas de ese prefijo, las máscaras, la disponibilidad y las huellas. Este recuento parcial no acredita los 5.023 activos. El [recibo](../../reports/data/historical-encoding-progress-20261008.json) conserva ambas ediciones y el contraste de tamaños de lote que precede a la ampliación.

La ruta de [tabla de palabras en CPU](frozen-embedding-placement.md) materializa otra edición uniforme y reduce la VRAM requerida en las formas comprobadas. Su primer bloque tiene 64 activos US y 230.957 muestras verificadas. El [contraste completo de ese bloque](../../reports/data/historical-cpu-encoding-20261008.json) conserva columnas no numéricas iguales y vectores idénticos bit a bit a la edición anterior. Los 480 activos anteriores mantienen su identidad y no se incorporan como si hubiesen sido calculados por la ruta nueva. La nueva supervisión deberá identificar el manifiesto codificado y sus factores efectivos. La paridad de embeddings no deriva ni verifica las etiquetas.

El [avance verificado del 9 de octubre](../../reports/data/historical-cpu-encoding-progress-20261009.json) amplía esa edición nueva a 512 activos US y 1.773.121 muestras, con 6.055.749.273 bytes de Parquet y 264.189 entradas de noticias en caché. Se han comprobado todas las filas, máscaras, disponibilidades y huellas del prefijo. El pico Torch sigue en 163.579.392 bytes. Los ocho bloques procesan activos distintos, por lo que sus tiempos no son una comparación de velocidad. La paridad entre ediciones sigue acreditada para los primeros 64 activos, sin extrapolarla al resto. Los recuentos de v2 y v3 se solapan y no se suman.

## Edición v3 completa y objetivos residuales

La edición `encoded-history-v3-cpu-words`, con la política `historical_masked_2000_v1`, terminó de codificarse el 9 de octubre con el runtime `4d88b243`. El [recibo](../../reports/data/historical-edition-v3-targets-20261009.json) resume los manifiestos, la verificación independiente y el recuento de presencia por modalidad, con sus huellas y sin rutas locales.

| Elemento | US | CN | Total |
| --- | ---: | ---: | ---: |
| Activos codificados | 4.213 | 810 | 5.023 |
| Muestras | 14.407.908 | 2.668.116 | 17.076.024 |
| Filas con noticias | 18,6 % | 13,7 % | 17,9 % |
| Filas con fundamentales | 40,6 % | 0,8 % | 34,4 % |
| Objetivos de entrenamiento (hasta 2022) | 13.137.025 | 2.423.172 | 15.560.197 |
| Objetivos de validación (2023) | 1.027.173 | 194.649 | 1.221.822 |

De los 5.676 candidatos, 653 no tienen los precios necesarios. Quince activos US codificados no tienen ninguna muestra, así que los objetivos cubren 5.008 activos. Precios y gráficos son obligatorios en cada fila con la política con máscaras. El bit macro está activo en todas las filas porque cada decisión tiene al menos un indicador admisible, lo que no significa que estén observados los 140: cada posición conserva su propia máscara y su antigüedad. Las muestras coinciden exactamente con las 17.076.024 ventanas del [censo](#ventanas-de-entrada-y-condiciones-del-objetivo). La verificación independiente de la edición recorrió los 5.023 activos en 658 s y terminó con `corpus_complete` verdadero, `training_ready` falso y el test sin abrir. Los Parquet de muestras ocupan 57.521.573.389 bytes.

Los objetivos (`targets-v3`) se generaron con el motor NumPy en 1.201 s, con el commit `3ad472ec` de [#413](https://github.com/GonxKZ/mars-titan/pull/413). Esa corrección convierte a texto la sesión de los precios de EE. UU., que Parquet guarda como diccionario y pandas lee como categoría sin orden. El primer intento había fallado en el primer activo por ese motivo. Se excluyen 294.005 filas, cada una con su motivo:

| Motivo | Filas |
| --- | ---: |
| Historia insuficiente para estimar el residual | 265.880 |
| Sesión siguiente ausente | 18.237 |
| Objetivo posterior al corte de 2023 | 4.962 |
| Objetivo que cruza la frontera entre entrenamiento y validación (purgado) | 4.926 |

Las filas aceptadas y las excluidas suman exactamente las muestras de la edición. La verificación independiente comprobó cada activo contra su recibo, la alineación fila a fila con las muestras, las particiones y que cada objetivo madura después de su predicción y antes de 2024. Después recalculó 190.830 etiquetas de 60 activos elegidos con la semilla 20261009 (40 US y 20 CN) con la referencia `residual_targets` en pandas, distinta del motor NumPy, y la diferencia absoluta máxima fue 0,0. Los factores de mercado SPY y CSI 300 se declaran sin versiones contemporáneas verificadas (`point_in_time_verified` falso).

Ninguna de estas pasadas ajustó modelos ni abrió el año 2024. Al preparar las vistas aparecieron 25 archivos de muestras (22 US y 3 CN) que terminan con un grupo Parquet vacío, justo los activos cuyo número de muestras es múltiplo de 128. La edición no se reescribió. Los lectores los saltan desde [#416](https://github.com/GonxKZ/mars-titan/pull/416), con el mismo orden, lotes y cursores que el archivo sin grupos vacíos.

## Ventanas de entrada y condiciones del objetivo

El [censo de ventanas](../../reports/data/historical-window-census-20261008.json) concilia los 5.023 activos y sus 18.982.446 precios. Contiene 17.076.024 ventanas de 64 sesiones consecutivas. Este es el denominador de la geometría de entradas, no un recuento de objetivos residuales válidos ni de muestras ya codificadas.

Hay 16.162 ventanas sin precio en la sesión siguiente dentro del periodo y 4.962 en el corte final. Otra auditoría del prefijo encuentra 266.297 ventanas sin los 126 pares pasados observables que exige el residual. De ellas, 201.492 pertenecen a activos cuya primera ventana de esta copia es posterior a 2000. Esa fecha no se interpreta como fecha de salida a bolsa. Los grupos de exclusión no se suman como si fueran disjuntos.

El segundo recuento usa SPY de la preparación verificada y el factor CN v16, ambos limitados a 2023. Todavía debe fijarse la revisión efectiva del factor US para las etiquetas. No se afirma que sea idéntico al descriptor anterior. Se conserva la ventana de 252 sesiones, la disponibilidad anterior a la decisión y el umbral de varianza vigente. Se contrastaron 40.064 posiciones con el cálculo directo y 16 fixtures de umbral y sufijo futuro. No se calcularon alpha, beta ni etiquetas. La ausencia de revisiones históricas tampoco queda acreditada por estos controles.

Estas condiciones necesitan resoluciones distintas. La falta de historia puede comprobarse desde el prefijo. La ausencia retrospectiva del precio siguiente no acredita que se conociese al emitir. El consumidor debe avanzar con todos los inputs admitidos y resolver los casos sin fabricar etiquetas o utilizar ese motivo como entrada. Los recuentos son acumulados, no un pico medido de pendientes. La suficiencia de la cola y los cortes de cada fase siguen pendientes de conciliación.

## Tablas preparadas

| Mercado | Candidatos | Activos preparados | Sin precios | Filas de precios | Noticias admitidas | Hechos contables admitidos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 4.784 | 4.213 | 571 | 16.090.522 | 2.156.832 | 3.027.615 |
| CN | 892 | 810 | 82 | 2.891.924 | 553.392 | 0 |

Los precios US comienzan el 3 de enero de 2000 y los CN el 4 de enero de 2006. Ambos terminan el 29 de diciembre de 2023. Cada activo conserva su propio comienzo y sus huecos. La ausencia de noticias o cuentas no elimina el activo. Entre los activos US preparados, 1.537 carecen de fuente contable y 90 de fuente de noticias.

Los archivos contables CN contienen 202.767 registros sin publicación acreditada. Esta pasada los identifica y no los convierte en entradas conocidas en el pasado. El enriquecimiento separado descrito más abajo incorpora comunicados oficiales revisados sobre este padre. Los archivos originales permanecen intactos. Los hechos US repetidos se deduplican según el contrato existente. Se excluyen 9.224 registros cuyo periodo termina después de su presentación, conservando su procedencia y el resto válido de cada empresa.

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

La edición histórica de aprendizaje ya incorpora las cuentas revisadas y sigue pendiente de codificar todas las ventanas válidas y comprobar objetivos, máscaras, exclusiones y cortes comunes. Estos resultados no miden mejoras predictivas ni habilitan el entrenamiento.
