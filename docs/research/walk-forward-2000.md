# Validación walk-forward v2 de la edición desde 2000

Revisión del 9 de octubre de 2026. Este documento fija el protocolo temporal con el que se entrenarán y compararán todas las familias sobre la [edición histórica con máscaras](../data/historical-temporal-views.md). Incluye el [protocolo conjunto v3](#protocolo-conjunto-v3-de-la-campaña-a-v2), decidido ese mismo día para la campaña A v2. Describe un diseño y sus comprobaciones técnicas. No contiene resultados predictivos, no se ha ejecutado ningún entrenamiento con él y el test de 2024 sigue cerrado. Corresponde a la etapa 4 de la [campaña sobre la edición desde 2000](training-campaign-2000.md) y a [#363](https://github.com/GonxKZ/mars-titan/issues/363).

## Por qué hace falta otra versión

Los protocolos históricos actuales (`historical-masked-us-walk-forward.json` y `historical-masked-cn-walk-forward.json`) copian las diez ventanas mensuales de la campaña estricta y solo adelantan el comienzo del ajuste a 2000. Su primera validación empieza en diciembre de 2022. Toda la historia nueva entra únicamente como entrenamiento y la selección depende de dos meses de validación. Esos archivos no se modifican y conservan exactamente sus ventanas, como comprueban las pruebas de paridad.

La versión 2 recorre la historia con ventanas de evaluación anuales. Cada ventana entrena con todo el pasado disponible desde 2000, valida en el tramo final de ese pasado y evalúa el año siguiente completo. Así cada año desde el primero con historia suficiente cuenta como periodo fuera de muestra y la selección usa seis meses de validación.

## Cortes de cada ventana

Para un año evaluado Y, los tramos son intervalos cerrados por la izquierda y abiertos por la derecha:

| Tramo | Intervalo | Función |
| --- | --- | --- |
| Ajuste | [2000-01-01, (Y−1)-04-01) | Pesos, normalizadores y cualquier transformación ajustable |
| Validación | [(Y−1)-04-01, (Y−1)-10-01) | Selección de configuración, de época y parada |
| Calibración | [(Y−1)-10-01, Y-01-01) | Intervalos y abstención, para los métodos que la necesiten |
| Evaluación | [Y-01-01, (Y+1)-01-01) | Comportamiento fuera de muestra de la regla ya seleccionada |

El paso entre ventanas es de doce meses y la última evalúa 2023. Las fechas desde 2024 se marcan `test_reserved` en todas las ventanas. Los seis meses de validación y los tres de calibración son los que ya proponía el [protocolo](protocol.md#particiones-ajuste-y-calibración). El precio es que el ajuste termina nueve meses antes de la evaluación. Volver a ajustar con validación y calibración incluidas duplicaría el coste y dejaría la parada sin un tramo independiente, así que no se hace.

Las configuraciones son `configs/evaluation/historical-masked-us-walk-forward-v2.json`, `historical-masked-cn-walk-forward-v2.json` y `historical-masked-joint-us-walk-forward-v2.json`. Tienen `schema_version=2`, `purge="label_interval"` y un bloque `selection` con la regla de parada. Si una ventana se usa para decidir el modelo definitivo, forma parte del desarrollo y no sustituye al test final.

## Primer año evaluado

El primer año no depende de la fecha nominal de inicio sino de la primera etiqueta madura. El objetivo residual necesita 126 pares de rendimientos observados dentro de las últimas 252 sesiones. Con los primeros precios el 3 de enero de 2000 en US y el 4 de enero de 2006 en CN, la primera etiqueta posible cae en la sesión 126, el 30 de junio de 2000 en US y el 14 de julio de 2006 en CN. Los activos que empiezan después tienen su propio calentamiento y simplemente aportan filas más tarde.

El protocolo exige al menos tres años de etiquetas maduras antes de la primera validación, como establece el [protocolo de investigación](protocol.md#particiones-ajuste-y-calibración). La primera validación que cumple esa condición en abril es la de 2004 para US y la de 2010 para CN. `minimum_train_months` traduce la regla a meses nominales desde 2000 (42 en US y 114 en CN y en el protocolo conjunto), y una prueba comprueba que un año antes ya no se cumpliría.

| Brazo | Años evaluados | Ventanas | Archivos |
| --- | --- | ---: | --- |
| US | 2005 a 2023 | 19 | `historical-masked-us-walk-forward-v2.json` |
| CN | 2011 a 2023 | 13 | `historical-masked-cn-walk-forward-v2.json` |
| US+CN | 2011 a 2023 | 13 | `historical-masked-joint-us-walk-forward-v2.json` y `historical-masked-cn-walk-forward-v2.json` |
| US+CN v3 (campaña A v2) | 2005 a 2023, CN solo cuenta desde 2011 | 19 | `historical-masked-us-walk-forward-v2.json` y `historical-masked-joint-cn-walk-forward-v3.json` |

## Mercados sin filas en una ventana

Las ventanas conjuntas empiezan cuando los dos mercados tienen etiquetas en los cuatro tramos con la historia mínima. El contrato v2 no admite un mercado vacío. La preparación rechaza antes de publicar cualquier ventana con un tramo sin filas en algún mercado y nombra la ventana, el mercado y el tramo. La alternativa de admitir CN vacío con una ponderación por mercado declarada se descarta por cinco motivos:

1. En 2005 el brazo conjunto tendría exactamente las filas del brazo US, y en 2006 y 2007 a CN le faltaría al menos el tramo de ajuste. Costaría casi lo mismo que el brazo US y no diría nada sobre la agrupación de mercados.
2. Entre 2008 y 2010 CN tendría todos los tramos, pero con menos de tres años de ajuste propio. La diferencia mezclaría el efecto de agrupar con el de una historia insuficiente.
3. Con un mercado vacío la media entre mercados no está definida. Renormalizar el peso cambia el estimando de una ventana a otra sin que lo decida el diseño.
4. Los consumidores existentes exigen filas en todos los tramos (`has_all_partitions` y recuentos positivos). Mantener esa invariante evita cambiar código de otras áreas.
5. No se pierde historia de ajuste. Las ventanas conjuntas siguen entrenando con todas las filas US desde 2000 y todas las filas CN desde 2006.

La ponderación entre mercados de las ventanas que sí se evalúan se declara en [métricas](metrics.md) y es la misma para todos los modelos.

La comparación declara además un análisis secundario por presencia de noticias y fundamentales, fijado el 9 de octubre de 2026 antes de cualquier resultado. Es descriptivo, no interviene en la selección ni en la parada de ningún brazo y no cambia la métrica principal. Su definición está en [métricas](metrics.md#estratos-por-presencia-de-modalidades).

El mismo 9 de octubre de 2026, también antes de cualquier resultado, se declaró como análisis secundario la ablación de modalidades en inferencia: el estado elegido de cada brazo vuelve a predecir la evaluación con noticias, fundamentales o ambos ausentes, sin reentrenar ni recalibrar. Está definida en [métricas](metrics.md#ablación-de-modalidades-en-inferencia) y no se ha ejecutado.

Los protocolos US y conjunto comparten cortes. Las ventanas conjuntas coinciden fecha a fecha con las ventanas US de 2011 a 2023, aunque sus identificadores empiezan en `fold-000`. Una comparación entre el brazo US y el conjunto sobre filas US se hace con esas trece ventanas, emparejadas por el intervalo de evaluación y no por el identificador.

Esta sección describe el protocolo conjunto v2, que conserva la campaña A original. La campaña A v2 lo sustituye por el [protocolo conjunto v3](#protocolo-conjunto-v3-de-la-campaña-a-v2), que admite China vacía en las ventanas donde no cuenta.

## Protocolo conjunto v3 de la campaña A v2

### Decisión

El 9 de octubre de 2026, antes de cualquier resultado, el autor decidió cambiar el diseño de la campaña A. Todas las familias entrenan un único modelo conjunto US+CN con un protocolo nuevo que empieza como US, con la primera validación en abril de 2004 y 19 ventanas hasta 2023. China entra en el ajuste en cuanto tiene filas, pero sus métricas solo cuentan en las ventanas que cumplen su propia historia mínima. Tres familias entrenan además modelos separados de US y de CN como controles, para contrastar en cada mercado el modelo conjunto con el separado. La [campaña](training-campaign-2000.md#campaña-a-v2-con-modelo-conjunto) recoge el resto del diseño. Es una decisión de diseño previa a cualquier resultado, no una conclusión.

### Cortes y filas de China

`configs/evaluation/historical-masked-joint-cn-walk-forward-v3.json` es el protocolo US v2 con `market="CN"`. La unión exige que los dos protocolos solo difieran en el mercado, así que US+CN usa `historical-masked-us-walk-forward-v2.json` para US y este archivo para CN. Las ventanas conjuntas coinciden fecha a fecha con las de US, con los mismos identificadores.

Las filas de China en las seis primeras ventanas salen de los objetivos con la misma regla que las vistas (ajuste, validación, calibración y evaluación):

| Ventana | Validación desde | Filas CN | Cuenta para CN |
| --- | --- | --- | --- |
| `fold-000` | 2004-04 | 0 / 0 / 0 / 0 | No |
| `fold-001` | 2005-04 | 0 / 0 / 0 / 46.902 | No |
| `fold-002` | 2006-04 | 0 / 22.097 / 24.394 / 103.042 | No |
| `fold-003` | 2007-04 | 70.136 / 52.657 / 26.715 / 112.310 | No |
| `fold-004` | 2008-04 | 177.275 / 56.079 / 28.845 / 118.266 | No |
| `fold-005` | 2009-04 | 290.577 / 61.352 / 28.969 / 122.072 | No |
| `fold-006` a `fold-018` | 2010-04 a 2022-04 | Las mismas que `fold-000` a `fold-012` de CN v2 | Sí |

China aporta filas de ajuste desde `fold-003` y de validación desde `fold-002`. En las ventanas que no cuentan para China, esas filas entran en el ajuste y en la selección por validación del modelo conjunto, porque son pasado disponible. La selección mira el MAE de validación de todas las filas de la vista, como en cualquier otro ámbito.

### Elegibilidad por mercado

Una ventana conjunta cuenta para un mercado si sus cuatro tramos son idénticos a los de alguna ventana del protocolo propio de ese mercado (`splits.eligible_folds`). Para China el protocolo propio es CN v2, con 114 meses de historia mínima, así que cuentan `fold-006` a `fold-018` (2011 a 2023). US cuenta en las 19.

Las predicciones de China en las ventanas no elegibles se generan y se conservan, pero quedan fuera de la calibración común y de todas las métricas (`predicted_and_kept_excluded_from_calibration_and_metrics`). La comparación cuenta las filas excluidas en cada ventana, brazo y tramo (`excluded_rows`). Se descartó marcarlas y evaluarlas aparte, porque serían años con menos de tres años de etiquetas de China y mezclarían el efecto de agrupar con el de una historia insuficiente, el segundo motivo del apartado anterior.

La preparación admite tramos vacíos de un mercado solo en sus ventanas no elegibles y los declara en el informe (`empty_folds_allowed`). `has_all_partitions` exige filas en los cuatro tramos de cada mercado elegible en la ventana, y el informe de la unión guarda los mercados elegibles de cada ventana y la huella del protocolo que fija la elegibilidad. Así se responden los motivos anteriores: el contraste con los controles separados mide el efecto de agrupar, China no cuenta con historia insuficiente, la media entre mercados no se renormaliza porque las filas de China excluidas no entran en ninguna métrica, los consumidores siguen exigiendo filas donde se mide y no se pierde historia de ajuste.

La vista agregada US+CN mezcla años solo con US (2005 a 2010) y años con los dos mercados. Por eso el contraste principal del diseño se hace por mercado, con las vistas US y CN del ámbito conjunto, y la vista agregada es descriptiva.

### Comparación con los controles separados

La [comparación conjunta](../../configs/evaluation/historical-masked-2000-joint-comparison.json) tiene versión 4 y un bloque `joint_design`. El ámbito US+CN compara todos los brazos. Los ámbitos US y CN comparan cada control separado (`transformer_compact`, `titans_mac_online` y `mars_titan_m1`) con el mismo brazo del modelo conjunto restringido a las filas de ese mercado, publicado como brazo prestado `<brazo>_joint`. La familia `joint_vs_separate` declara el delta del conjunto menos el separado, con el separado como base, y hereda el remuestreo por bloques de días y el máximo estudentizado de las demás familias.

Las ventanas se emparejan por intervalos idénticos: `fold-k` de US con `fold-k` del conjunto y `fold-k` de CN con `fold-(k+6)`. Las filas y los objetivos del brazo prestado deben coincidir exactamente con los del control, y la huella de la vista conjunta de cada ventana queda en el manifiesto de fuentes. Cada semilla se resume por separado y los contrastes usan la media sesión a sesión de las semillas. Ridge y XGBoost tienen una sola semilla y entran con su serie.

## Purga por el intervalo de cada etiqueta

Una fila pertenece a un tramo solo si su decisión está dentro del tramo y su etiqueta madura estrictamente antes del final, es decir, si el intervalo [`prediction_at`, `label_available_at`] cabe entero en [inicio, fin). El objetivo es el residual apertura-cierre de la sesión siguiente. Su entrada (`entry_at`) y su salida (`exit_at`) son la apertura y el cierre de esa sesión, posteriores a la decisión, y `label_available_at` es la última disponibilidad del activo y del factor. La tabla de etiquetas no guarda `entry_at` ni `exit_at`, pero ambos quedan dentro del intervalo usado, que es por tanto una cota conservadora de la purga.

La regla se aplica igual en ajuste, validación, calibración y evaluación. Una etiqueta que madura exactamente en la frontera se purga. La versión 2 elimina el margen fijo de una sesión. Para etiquetas de la sesión siguiente ese margen retiraba las mismas filas que la purga por intervalo, solo con otro motivo (`session_gap`), y una prueba lo comprueba en las diecinueve ventanas. La regla nueva no depende del horizonte y sigue siendo correcta si una etiqueta madura más tarde.

Las filas purgadas conservan el motivo `label_crosses_boundary`. Además, cada activo, cada vista y cada informe registran `purged_by_boundary` con el número de filas que cruzan el final de cada tramo, y la unión conjunta lo desglosa por mercado. Las entradas posteriores a la decisión, las incompletas y los motivos de exclusión del padre se conservan como antes. La etiqueta anual recuperada del corte de 2022 a 2023 nunca se admite en la versión 2, porque siempre queda al final de un tramo.

## Regla común de selección y parada

El bloque `selection` es idéntico en los tres archivos y se fija antes de preparar las vistas, igual que las semillas y la métrica primaria:

| Campo | Valor | Significado |
| --- | ---: | --- |
| `metric` | `session_mae` | MAE residual por sesión en el tramo de validación |
| `stopping` | `fixed_budget` | Se recorren todas las épocas y se conserva el mejor estado |
| `max_epochs` | 30 | Número de evaluaciones completas de cada ajuste |
| `patience` | 5 | Diagnóstico `plateau_epoch`, la época en la que habría parado una meseta |
| `min_delta` | 0,00001 | Mejora estricta necesaria para sustituir el mejor estado |

Son los valores de la [búsqueda histórica de referencias](../../configs/baselines/historical-masked-reference-search-us.json) (versión 4), que conserva las épocas, la paciencia y la mejora mínima de la [búsqueda temporal estricta](../engineering/strict-temporal-search.md) y sustituye su meseta por el presupuesto fijo. Con la misma población, el mismo lote y las mismas épocas, todos los brazos aplican el mismo número de actualizaciones, como exige la comparación emparejada. El sobreajuste se controla con la selección del mejor estado de validación, no cortando el presupuesto. Esta selección reduce el riesgo de quedarse con un estado sobreajustado, pero no lo elimina.

El esquema admite también `stopping="validation_plateau"` con `minimum_epochs`, que es el criterio de [#190](https://github.com/GonxKZ/mars-titan/issues/190). No se usa aquí porque haría que cada brazo se detuviera con un número distinto de actualizaciones, y eso sería otra comparación. Cambiar la regla exige publicar otro archivo con otra identidad antes de entrenar.

La semántica es la del [selector existente](../engineering/masked-reference-runners.md#presupuesto-fijo-y-selección). Solo cuentan las evaluaciones completas, un empate no mejora, una mejora menor que `min_delta` tampoco, y una continuación incluye el padre como época 0 elegible. El estado del selector es un registro serializable, así que reanudar a mitad de la paciencia conserva la decisión. Las pruebas recorren secuencias de métricas fijadas con los dos modos, sin modelo ni pasos de optimizador.

La búsqueda temporal compara la regla del plan con la del protocolo, incluidos el modo y el mínimo, y rechaza cualquier diferencia. Los planes de versiones 2 y 3 equivalen a una meseta, así que solo sirven para los protocolos v1. Los ejecutores de Titans-MAC, MARS-TITAN y CM-v1 deben leer la regla con `stopping_rule(protocol)` cuando se conecten a estas vistas. Una parada independiente que cambie el número de actualizaciones se registra como otra comparación.

## Estado y calentamiento por ventana

Cada ventana parte de una inicialización nueva con su semilla. No hereda pesos, optimizador, normalizadores ni calibrador de otra ventana. Los pesos rápidos y el momentum de Titans, el banco episódico, la cola de etiquetas pendientes y el cursor se reinician al empezar cada ventana y al empezar cada pasada de validación, calibración o evaluación.

El calentamiento de los brazos con memoria usa solo las sesiones anteriores al comienzo del tramo que se va a medir y solo etiquetas con `label_available_at` anterior a ese comienzo. Su longitud será la misma para todos los brazos con memoria. Esta política está declarada aquí y su aplicación corresponde a cada ejecutor. El [punto de entrada de Titans-MAC](../engineering/titans-chronological-trainer.md#ventana-walk-forward) la aplica con 12 meses de entradas, sin etiquetas, limitados al origen del ajuste. Los demás ejecutores todavía no la conectan a estas vistas. La separación entre el checkpoint seleccionado y los de recuperación, con rotación acotada, sigue la [política de checkpoints](../engineering/checkpoint-recovery.md) y se coordina con [#67](https://github.com/GonxKZ/mars-titan/issues/67).

## Ventanas y filas nominales

La tabla da las fechas de cada ventana, las sesiones de validación, calibración y evaluación por mercado (calendarios XNYS y XSHG de `exchange_calendars`) y las filas nominales de ajuste en millones. Las filas nominales reparten por sesiones las ventanas de 64 precios de cada año del [censo de ventanas](../../reports/data/historical-price-windows-20261007.json). Son una cota superior, no objetivos válidos. Faltan los descuentos por calentamiento, sesión siguiente ausente y purga.

| Año evaluado | Fin del ajuste | Calibración | Sesiones US | Sesiones CN | Ajuste US | Ajuste CN | Ajuste US+CN |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 2005 | 2004-04-01 | 2004-10-01 | 126 / 64 / 252 | No evaluado | 1,13 | | |
| 2006 | 2005-04-01 | 2005-10-01 | 128 / 63 / 251 | No evaluado | 1,46 | | |
| 2007 | 2006-04-01 | 2006-10-01 | 126 / 63 / 251 | No evaluado | 1,83 | | |
| 2008 | 2007-04-01 | 2007-10-01 | 126 / 64 / 253 | No evaluado | 2,23 | | |
| 2009 | 2008-04-01 | 2008-10-01 | 128 / 64 / 252 | No evaluado | 2,66 | | |
| 2010 | 2009-04-01 | 2009-10-01 | 127 / 64 / 252 | No evaluado | 3,12 | | |
| 2011 | 2010-04-01 | 2010-10-01 | 127 / 64 / 252 | 123 / 61 / 244 | 3,61 | 0,44 | 4,06 |
| 2012 | 2011-04-01 | 2011-10-01 | 127 / 63 / 250 | 126 / 60 / 243 | 4,14 | 0,57 | 4,71 |
| 2013 | 2012-04-01 | 2012-10-01 | 126 / 62 / 252 | 124 / 61 / 238 | 4,69 | 0,71 | 5,40 |
| 2014 | 2013-04-01 | 2013-10-01 | 128 / 64 / 252 | 121 / 61 / 245 | 5,26 | 0,86 | 6,12 |
| 2015 | 2014-04-01 | 2014-10-01 | 127 / 64 / 252 | 126 / 61 / 244 | 5,87 | 1,01 | 6,88 |
| 2016 | 2015-04-01 | 2015-10-01 | 127 / 64 / 252 | 126 / 61 / 244 | 6,54 | 1,16 | 7,70 |
| 2017 | 2016-04-01 | 2016-10-01 | 128 / 63 / 251 | 125 / 60 / 244 | 7,25 | 1,32 | 8,57 |
| 2018 | 2017-04-01 | 2017-10-01 | 126 / 63 / 251 | 125 / 60 / 243 | 8,01 | 1,48 | 9,50 |
| 2019 | 2018-04-01 | 2018-10-01 | 127 / 63 / 252 | 124 / 60 / 244 | 8,81 | 1,66 | 10,47 |
| 2020 | 2019-04-01 | 2019-10-01 | 127 / 64 / 253 | 125 / 61 / 243 | 9,67 | 1,82 | 11,48 |
| 2021 | 2020-04-01 | 2020-10-01 | 127 / 64 / 252 | 125 / 60 / 243 | 10,60 | 1,95 | 12,54 |
| 2022 | 2021-04-01 | 2021-10-01 | 127 / 64 / 251 | 124 / 61 / 242 | 11,58 | 2,13 | 13,71 |
| 2023 | 2022-04-01 | 2022-10-01 | 126 / 63 / 250 | 124 / 60 / 242 | 12,60 | 2,33 | 14,93 |
| Suma | | | | | 111,1 | 17,4 | 116,1 |

La validación empieza en la fecha de fin del ajuste y la evaluación el 1 de enero del año evaluado. Las filas nominales de validación por ventana están entre 0,16 y 0,52 millones en US, así que evaluar la validación en cada época añade alrededor de un 5 % de pasadas hacia delante que la cuenta siguiente no incluye.

## Coste y alternativas de presupuesto

La auditoría de preparación supone entre 15 y 22 minutos por época de una referencia GRU sobre unos 15,5 millones de filas. Es una hipótesis basada en caudales de ediciones anteriores, no una medida sobre esta edición. Equivale a entre 58 y 85 microsegundos por fila y época. Con presupuesto fijo, el coste de entrenar una familia con una configuración y una semilla es aproximadamente la suma, para cada ventana, de filas de ajuste por épocas por ese tiempo. Las cifras usan las filas nominales de la tabla y por eso sobrestiman algo.

| Alternativa | US | CN | US+CN |
| --- | ---: | ---: | ---: |
| A. Reentrenar todas las ventanas desde cero, 30 épocas | 53,7 a 78,8 h | 8,4 a 12,4 h | 56,2 a 82,4 h |
| B. Reentrenar desde cero cada tres años (US 7 ventanas, CN y US+CN 5), 30 épocas | 20,2 a 29,6 h | 3,3 a 4,8 h | 21,9 a 32,0 h |
| B. Reentrenar desde cero cada dos años (US 10 ventanas, CN y US+CN 7), 30 épocas | 28,6 a 41,9 h | 4,6 a 6,7 h | 30,4 a 44,6 h |
| C. Primera ventana completa con 30 épocas y continuación con 5 millones de filas por ventana | 2,0 a 2,9 h | 1,2 a 1,7 h | 2,9 a 4,3 h |
| D. Cada ventana desde cero con un presupuesto fijo de 20 millones de filas | 6,1 a 9,0 h | 4,2 a 6,2 h | 4,2 a 6,2 h |

Las cifras de A son coherentes con las 85 horas que estimó la auditoría para unas 19 ventanas y unos 130 millones de filas acumuladas. Con una meseta de mínimo 3 y paciencia 5, una ejecución plana se detendría en la época 8 y A bajaría como mínimo a unas 14 a 21 horas en US, a cambio de perder la igualdad de actualizaciones. Para toda la campaña hay que multiplicar por brazos y semillas. Con 20 brazos neuronales y tres semillas, una cifra solo ilustrativa, A supondría entre 3.200 y 4.700 horas en US, B cada tres años entre 1.200 y 1.800, C entre 120 y 175 y D entre 370 y 540. El caudal de Titans-MAC, MARS-TITAN y CM-v1 sobre esta edición no se ha medido y puede ser menor que el de la GRU.

El plan histórico de referencias ejecuta 50 trabajos por ventana. Con 19 ventanas serían 950 y con 13 serían 650, por encima del límite predeterminado de 512 que aplica `temporal_search`. B cada tres años los deja en 350 y 250. Un plan de versión 4 puede declarar otro límite con `max_runs`, entre 1 y 4.096, que queda en la huella del plan.

Cada alternativa responde a una pregunta algo distinta:

- **A** es la referencia del diseño. Cada ventana es un ajuste independiente que solo ve su pasado.
- **B** conserva esa semántica en un subconjunto fijado ahora, sin mirar resultados. Se declara con un protocolo v2 cuyo `step_months` es 36 o 24. Con paso 36, las ventanas conjuntas (2011, 2014, 2017, 2020 y 2023) son un subconjunto exacto de las ventanas US (2005, 2008, 2011, 2014, 2017, 2020 y 2023). La [variante B de la campaña](training-campaign-2000.md#variantes-de-presupuesto) reentrena esas mismas ventanas, pero evalúa también los años intermedios con el estado de la última ventana reentrenada, sin ajustarlo.
- **C** parte en cada ventana del estado seleccionado en la anterior, que solo ha visto datos previos a su propia validación. No hay fuga, pero el resultado de una ventana depende del camino anterior y las ventanas dejan de ser ajustes independientes. Las memorias se siguen reiniciando por ventana. El padre debe ser elegible como época 0 y el presupuesto de continuación es igual para todos los brazos. Es otra comparación y debe presentarse como tal.
- **D** mantiene ajustes independientes con el mismo número de actualizaciones en todas las ventanas, muestreando de toda la historia disponible. En la última ventana US equivale a 1,6 pasadas y en las primeras de CN a muchas más, así que su valor debe fijarse por comparación.

La orientación inicial era B con paso de 36 meses, que conserva la semántica del diseño con un tercio del coste de A y cabe en el límite de trabajos. El 9 de octubre de 2026 se eligió A con todas las familias, registrada en [#363](https://github.com/GonxKZ/mars-titan/issues/363), sin esperar al caudal medido: se prefirió que cada ventana entrene con todo su pasado disponible. Las cifras de coste de esta sección siguen siendo hipótesis. La duración real se medirá con los primeros trabajos y se registrará en la misma tarea, sin cambiar a B ni recortar filas en silencio.

## Uso

La preparación necesita la supervisión histórica verificada. La edición v3 y sus objetivos residuales se verificaron el 9 de octubre ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)) y las [vistas de la campaña A](#vistas-de-la-campaña-a) se prepararon y verificaron ese mismo día con `run_masked_campaign.py prepare`, descrito en el [plan de la campaña](training-campaign-2000.md#ejecución-y-recuperación). Para unas vistas conjuntas sueltas:

```bash
uv run --no-sync python -m mars_titan.training.joint_temporal_corpus \
  --parent <supervisión histórica> --output <vistas conjuntas v2> \
  --us-protocol configs/evaluation/historical-masked-joint-us-walk-forward-v2.json \
  --cn-protocol configs/evaluation/historical-masked-cn-walk-forward-v2.json \
  --input-policy historical_masked_2000_v1 --recover-annual-boundaries

uv run --no-sync python -m mars_titan.training.temporal_search \
  --config <plan de referencias de versión 4> --views <vistas conjuntas v2> --check
```

Las vistas conjuntas v3 se preparan con la campaña A v2, que aplica la elegibilidad de China. No se han generado. Ocuparían unos 8,3 GB y tardarían unos 76 minutos, por proporción con las 13 ventanas de la v2 (5,7 GB y 52 minutos). La preparación espera a la confirmación del autor y a la revisión del hueco de datos chinos entre mayo y julio de 2019, que se ve en la validación de `fold-015` (40.605 filas CN frente a unas 90.000 en las ventanas vecinas):

```bash
uv run --no-sync python scripts/run_masked_campaign.py prepare \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --parent <supervisión histórica>/targets-v3/manifest.json \
  --output <vistas conjuntas v3> --scope US+CN
uv run --no-sync python scripts/run_masked_campaign.py views \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --views US+CN=<vistas conjuntas v3>/US+CN --views US=<vistas>/US --views CN=<vistas>/CN
```

Las vistas US y CN de la campaña A sirven sin cambios para los controles separados. La comprobación de vistas de la campaña A v2 las acepta con sus 19 y 13 ventanas. Un verificador local independiente, adaptado del de la campaña A, recalcula las fronteras, comprueba fila a fila la purga y los objetivos, recalcula la elegibilidad de China frente a la declarada y compara cada ventana conjunta con su gemela por mercado (US v2 en las 19 y CN v2 en las elegibles).

`--check` valida informes, ventanas, población, política y regla de parada, y devuelve la identidad y el número de trabajos previstos sin reservar la GPU ni entrenar. La búsqueda lee la política de entradas del plan, como hace la búsqueda de referencias, y exige que las vistas declaren la misma. Un plan estricto no puede leer vistas con máscaras ni al revés. El plan debe usar `arms=["US+CN"]` con las vistas conjuntas y el mercado correspondiente con las vistas de un solo mercado.

## Vistas de la campaña A

Las vistas reales de la campaña A se prepararon el 9 de octubre sobre los objetivos `targets-v3`, con la [configuración de A](../../configs/baselines/historical-masked-campaign-a.json) y el runtime `7a9e93e3`. Cada ámbito se preparó en un proceso de un hilo que repite el cuerpo del bucle de `prepare_views`, y los tres procesos terminaron con código 0. Después, `run_masked_campaign.py prepare` con la campaña A sobre `targets-v3/manifest.json` validó los tres destinos con su comprobación oficial y también terminó con código 0. El [recibo](../../reports/data/campaign-a-views-20261009.json) conserva las huellas de los informes, los recuentos de cada ventana, las filas purgadas y los tiempos.

| Ámbito | Ventanas | Entrenamiento | Validación | Calibración | Evaluación | Purgadas en fronteras | Preparación |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 19 | 108.111.890 | 6.084.968 | 3.087.269 | 12.826.460 | 194.849 | 50 min |
| CN | 13 | 16.903.888 | 1.015.019 | 525.177 | 2.105.339 | 34.300 | 16 min |
| US+CN | 13 | 113.180.071 | 5.899.203 | 2.999.327 | 12.357.256 | 189.557 | 52 min |

Los recuentos suman todas las ventanas de cada ámbito, así que una fila cuenta una vez por cada ventana que la usa. Las filas de entrenamiento quedan alrededor del 97 % de la cota nominal de la [tabla anterior](#ventanas-y-filas-nominales) (111,1, 17,4 y 116,1 millones). La evaluación de la última ventana, el año 2023, coincide en los tres ámbitos con las filas de validación de `targets-v3`: 1.027.173 en US, 194.649 en CN y 1.221.822 en US+CN. Las vistas ocupan 12,9 GB de datos y 13,9 GB en disco contando sus directorios. Los tiempos son de reloj, con los tres procesos ejecutándose a la vez.

Una verificación independiente, escrita sin reutilizar el código que preparó las vistas, recorrió los 5.008 activos con objetivos y las 45 ventanas. Comprobó:

- las fronteras de cada ventana, recalculadas desde los protocolos, y la purga por el intervalo de cada etiqueta,
- que ningún objetivo queda fuera de su ventana y que ninguna decisión ni maduración llega a 2024,
- que cada objetivo es idéntico bit a bit al de `targets-v3`,
- que no se pierde ninguna fila elegible y que los recuentos coinciden con los manifiestos de cada ventana,
- que US+CN reproduce exactamente las vistas por mercado con las mismas fronteras.

Terminó sin fallos en 8 minutos con cinco procesos. Las vistas, el verificador y su informe son locales, y el recibo guarda la huella de los dos últimos. Ninguna de estas pasadas ajustó modelos ni abrió 2024. La campaña no se ha lanzado.

## Comprobaciones técnicas

Las pruebas de `tests/evaluation/test_walk_forward_v2.py` comprueban los límites de cada ventana, el primer año con tres años de etiquetas, la coincidencia entre ventanas conjuntas y US, la purga en cada frontera, la frontera exacta, la ausencia de filas de 2024 en tramos de desarrollo, la invariancia al cambiar o reordenar el sufijo futuro, la equivalencia con el margen anterior para etiquetas de la sesión siguiente, la regla de parada con secuencias fijadas en los dos modos y la paridad de los seis protocolos v1 mediante huellas calculadas antes del cambio.

Las pruebas de `tests/training/test_walk_forward_v2_views.py` preparan vistas sobre un corpus técnico con filas en todos los años, desde febrero de 2000 en US y julio de 2006 en CN. Comparan los recuentos por ventana, tramo, mercado y año con una derivación independiente, comprueban `purged_by_boundary`, el rechazo de una ventana conjunta con CN vacío, la comprobación de la búsqueda sin GPU, la lectura de la política desde un plan de versión 4, el rechazo de planes estrictos o con otra parada y la paridad de las vistas v1. La invariancia del objetivo residual ante cambios futuros ya la cubre `tests/data/test_budget_targets.py`.

Las pruebas de `tests/training/test_campaign_a_joint.py` comprueban la elegibilidad por intervalos idénticos, la unión con China vacía solo donde no cuenta y el rechazo en el resto, la exclusión de las filas no elegibles en calibración y evaluación con sus recuentos, el emparejamiento de los brazos prestados con su ventana y su vista y el rechazo de declaraciones ambiguas del diseño conjunto.

Nada de esto ejecuta modelos, pasos de optimizador ni evaluaciones científicas, ni genera objetivos reales. Los recuentos reales por ventana y mercado están en el [recibo de las vistas de la campaña A](../../reports/data/campaign-a-views-20261009.json).
