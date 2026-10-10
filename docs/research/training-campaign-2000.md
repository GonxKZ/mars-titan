# Campaña de entrenamiento sobre la edición histórica desde 2000

Revisión del 9 de octubre de 2026. Este documento ordena el recorrido completo desde los datos hasta la comparación final. Describe un plan y el estado de cada etapa. No contiene resultados predictivos: ningún modelo se ha entrenado sobre esta edición y el test de 2024 sigue cerrado.

## Objetivo

Entrenar desde cero todas las arquitecturas comparadas con todos los datos utilizables desde 2000, con las mismas filas, la misma validación temporal y presupuestos comparables. Después se aplican postentrenamientos y políticas de refuerzo sobre esas mismas bases. La pregunta es cuánto cambia el error de cada variante frente a sus referencias, con incertidumbre temporal y costes medidos.

La condición del bloqueo de aprendizaje era preparar la edición desde 2000 con ausencias y máscaras explícitas y la misma población para todos los modelos. La edición y sus objetivos residuales se verificaron el 9 de octubre ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)). Ese mismo día se eligió la [variante A](#variantes-de-presupuesto) con todas las familias y se prepararon y verificaron sus [vistas walk-forward](walk-forward-2000.md#vistas-de-la-campaña-a) ([recibo](../../reports/data/campaign-a-views-20261009.json)). La protección local sigue activa y la campaña no se ha lanzado. No se entrenará ningún modelo hasta terminar el código y las implementaciones pendientes y revisar un resumen de su estado. Ese mismo 9 de octubre, antes de cualquier resultado, el diseño de A cambió a un [modelo conjunto US+CN desde 2004 con controles separados](#campaña-a-v2-con-modelo-conjunto), declarado como campaña A v2 sin modificar la configuración original.

## Etapas y estado

| Etapa | Contenido | Estado a 9 de octubre | Tarea |
| --- | --- | --- | --- |
| 1. Edición de entradas | Codificación v3 de precios, noticias, gráficos, fundamentales y 140 posiciones macro con nivel, presencia y antigüedad, más cinco bits de presencia por modalidad | Completa. 5.023 activos codificados (4.213 US y 810 CN), 15 sin muestras, y 17.076.024 muestras. Precios, gráficos y macro en todas las filas, fundamentales en el 34,4 % y noticias en el 17,9 % ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)) | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 2. Objetivos | Retorno residual apertura-cierre de la sesión siguiente, OLS con 252 sesiones y un mínimo de 126 pares, solo con pares disponibles en la decisión | Generados y verificados. 15.560.197 filas de entrenamiento hasta 2022 y 1.221.822 de validación en 2023, con 294.005 exclusiones por motivo. El recálculo independiente de 190.830 etiquetas de 60 activos no encontró diferencias. La lectura de sesiones guardadas como diccionario se corrigió en [#413](https://github.com/GonxKZ/mars-titan/pull/413) | [#15](https://github.com/GonxKZ/mars-titan/issues/15) |
| 3. Verificación de la edición | Conciliación de los 5.676 candidatos, recuentos frente al censo de ventanas, máscaras, disponibilidad no posterior a la decisión, ninguna fila de 2024 | Hecha. La verificación independiente termina con `corpus_complete` verdadero y las filas aceptadas y excluidas suman exactamente las muestras de la edición. 2024 sigue fuera | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 4. Protocolo temporal | Ventanas anuales expansivas desde el primer año con etiquetas, validación interna al final de cada tramo de entrenamiento, evaluación del año siguiente, purga por intervalo real de etiquetas | Diseño y comprobaciones técnicas en el [protocolo v2](walk-forward-2000.md). Variantes de presupuesto A y B declaradas con su [recuento de trabajos](#variantes-de-presupuesto). La [orden de medición](#medición-de-caudal) cubre referencias, Titans-MAC, GRU candidata, lectores de MARS-TITAN, núcleos y lectores de CM-v1 y adaptadores, con una [declaración preparada](#declaración-preparada-de-las-familias-pendientes) de las tres familias que A y B no declaran. Variante A elegida. Sus [vistas reales](walk-forward-2000.md#vistas-de-la-campaña-a) están preparadas y verificadas de forma independiente: 19 ventanas en US y 13 en CN y en US+CN, con objetivos idénticos a `targets-v3` y ninguna fila elegible perdida. Los lectores saltan los grupos Parquet vacíos de 25 archivos de muestras ([#416](https://github.com/GonxKZ/mars-titan/pull/416)). La orden de caudal no se ha ejecutado. La [campaña A v2](#campaña-a-v2-con-modelo-conjunto) declara el protocolo conjunto v3 desde 2004, con sus vistas todavía sin generar | [#363](https://github.com/GonxKZ/mars-titan/issues/363) |
| 5. Entrenamiento base | Referencias, GRU episódica, Transformer compacto, núcleo Titans-MAC, MARS-TITAN con ampliaciones y CM-v1 | [Orquestación](#orquestación-de-los-brazos-con-entrenador) de las referencias neuronales y tabulares preparada y comprobada con dobles, sin ejecutar. Los entrenadores cronológicos de [Titans-MAC](../engineering/titans-chronological-trainer.md) y de la [GRU episódica](../engineering/candidate-chronological-trainer.md) están comprobados sin pasos de optimizador. Los dos tienen entrada por ventana y traslado para B, registrados en la orquestación mediante secciones opcionales y comprobados en CPU. Las campañas A y B declaran Titans-MAC con su [receta de campaña](../engineering/titans-chronological-trainer.md#receta-de-la-campaña-y-casos-de-búsqueda), sin ejecutar. La GRU episódica, MARS-TITAN y CM-v1 están en la declaración ampliada de A, todavía sin copiar a su configuración. Las [comprobaciones CUDA](../../reports/engineering/cuda-checks-20261009/README.md) de estas rutas pasan sin pasos de optimizador. M2 lo hace con el enlace de `develop` desde [#418](https://github.com/GonxKZ/mars-titan/pull/418) | [#234](https://github.com/GonxKZ/mars-titan/issues/234), [#23](https://github.com/GonxKZ/mars-titan/issues/23), [#293](https://github.com/GonxKZ/mars-titan/issues/293) |
| 6. Postentrenamiento | Padre congelado, continuación supervisada, corrección residual y adaptadores, solos y combinados | Edición con máscaras, matriz con pinball para padres de cuantiles, modo matriz de la cola y [etapa por ventana](../engineering/masked-posttraining.md#etapa-por-ventana-de-la-campaña) comprobados hasta el paso del optimizador, sin ejecutar. La etapa está registrada en la orquestación como etapa posterior, con el subcomando `posttraining` de la orden única | [#364](https://github.com/GonxKZ/mars-titan/issues/364), [#128](https://github.com/GonxKZ/mars-titan/issues/128) |
| 7. Refuerzo | Variantes de PPO, KLPO prioritario y Double DQN sobre entornos auditados | Controladores implementados sin ejecutar pasos. [Etapa de políticas por ventana](#etapa-de-políticas-por-ventana) declarada para A y B en dos niveles (KLPO y referencias sobre todos los predictores de la campaña, y PPO y Double DQN sobre dos), registrada en la orquestación con el subcomando `rl` y comprobada en CPU con ejecutores sustitutos, sin ejecutar. Los brazos aprendidos se lanzan con `mars-titan-ppo` (esquema 4) y `mars-titan-klpo` sobre cintas reconstruidas, también chinas con sus reglas, comprobados en diagnóstico CPU sin pasos de optimizador ([#420](https://github.com/GonxKZ/mars-titan/pull/420)). Falta compilarlos con LibTorch CUDA y medir su rendimiento en `cuda:0`. Ningún ajuste RL se ha ejecutado | [#137](https://github.com/GonxKZ/mars-titan/issues/137), [#365](https://github.com/GonxKZ/mars-titan/issues/365) |
| 8. Evaluación | MAE residual por sesión y métricas secundarias con incertidumbre por bloques | Métricas, calibración común CQR y [evaluación walk-forward](metrics.md#evaluación-walk-forward-de-la-edición-desde-2000) implementadas con pruebas técnicas y [configuración declarada](../../configs/evaluation/historical-masked-2000-comparison.json). Sin aplicar a esta edición. Las referencias neuronales y tabulares ya escriben las predicciones por fila de calibración y evaluación, y la orquestación publica el manifiesto de fuentes de cada ámbito. Las demás familias tienen productor en la declaración ampliada. La comparación declara además, antes de cualquier resultado, [estratos por presencia de noticias y fundamentales](metrics.md#estratos-por-presencia-de-modalidades) como análisis secundario descriptivo ([#415](https://github.com/GonxKZ/mars-titan/pull/415)) | [#32](https://github.com/GonxKZ/mars-titan/issues/32), [#36](https://github.com/GonxKZ/mars-titan/issues/36) |

## Población y equidad

Todas las arquitecturas leen las mismas filas de la edición. Una fila existe aunque falten noticias, fundamentales o macro, porque la ausencia se marca y se rellena con cero de forma explícita. Ninguna variante puede descartar filas difíciles para mejorar su promedio.

Los bits de presencia forman parte de la entrada de todas las familias. Titans y la GRU candidata ya los recibían. Con la política con máscaras las referencias neuronales los fusionan con las modalidades y Ridge, HistGradientBoosting y XGBoost los reciben como cinco columnas más. Sin ese cambio, una diferencia de error podría deberse a que un modelo distingue un cero real de una ausencia y otro no.

La comparación estricta, que exige las cuatro modalidades completas y los 140 indicadores, se conserva como control separado. No se mezclan sus filas con las de la edición desde 2000.

## Validación temporal y sobreajuste

Cada ventana entrena con todo el pasado disponible hasta su corte, valida en el tramo final de ese pasado y evalúa el año siguiente. La purga elimina las filas cuya etiqueta madura después del corte. China no tiene etiquetas residuales antes de 2006 porque sus precios y el factor CSI300 empiezan ese año. Por eso las ventanas conjuntas del protocolo v2 empiezan en 2011, el primer año con tres años de etiquetas maduras en los dos mercados, y ninguna ventana admite un mercado vacío. Las ventanas solo de US evalúan desde 2005 y siguen entrenando con todo el pasado disponible. El [protocolo v2](walk-forward-2000.md) justifica la decisión. La campaña A v2 usa en su lugar el [protocolo conjunto v3](walk-forward-2000.md#protocolo-conjunto-v3-de-la-campaña-a-v2): 19 ventanas desde 2004 con China en el ajuste en cuanto tiene filas y en las métricas solo desde 2011.

La selección guarda el mejor estado según la validación temporal, con presupuesto fijo de 30 épocas, paciencia y mejora mínima declaradas en el protocolo. Los controles emparejados conservan el mismo número de actualizaciones. Si una parada independiente rompiera esa igualdad, se usa selección del mejor estado con presupuesto fijo. Los checkpoints de recuperación rotan con un límite pequeño y el mejor estado se guarda aparte, según [la política de checkpoints](../engineering/checkpoint-recovery.md).

Los brazos con memoria predicen cada tramo desde el estado inicial, tras observar las entradas de los 12 meses anteriores sin etiquetas. El calentamiento es el mismo para los cuatro controles de Titans-MAC y está declarado en sus recetas antes de entrenar. Ninguna etiqueta madura se escribe en la memoria rápida. La [política completa](../engineering/titans-chronological-trainer.md#política-de-memoria-en-inferencia) explica por qué no se encadena la memoria entre tramos.

El test de 2024 no participa en ninguna selección. Se abrirá una sola vez, con la configuración fijada, al final de la campaña.

## Familias entrenadas

| Familia | Identidad | Qué aísla |
| --- | --- | --- |
| Referencias | RNN, LSTM, GRU, DLinear, Transformer compacto, Ridge y XGBoost | Nivel de error sin memoria persistente |
| GRU episódica | Candidato con banco 128×256, referencia independiente | Memoria episódica sobre un codificador recurrente |
| Núcleo Titans-MAC | `transformer_direct`, `mac_disabled`, `mac_frozen`, `mac_online` | Cambio de codificador frente a memoria neuronal |
| MARS-TITAN con ampliaciones | Banco episódico, escritura M0 a M3, refinamientos K=1, 2 y 4, y las modificaciones del [documento de integración](system-integration.md), cada una desactivable | Aportación de cada ampliación, una cada vez |
| CM-v1 | B, B+C, B+M y B+C+M, con B igual a Titans-MAC `mac_online` con C en `disabled` y el lector M1 con K=1, según [el factorial](../experiments/mars_titan_cm_v1/factorial.md) | Control del radio numérico en el ajuste del núcleo y consolidación del banco del lector |

No se ejecuta el producto cartesiano de todas las ampliaciones. Primero se fija la base y después se estudia un mecanismo cada vez, como establece el [documento de integración](system-integration.md).

## Orquestación de los brazos con entrenador

La campaña se declara en dos configuraciones que solo difieren en el presupuesto: la [variante A](../../configs/baselines/historical-masked-campaign-a.json) y la [variante B](../../configs/baselines/historical-masked-campaign-b.json). Brazos, semillas, ámbitos, protocolos y ventanas se leen de la [comparación walk-forward declarada](../../configs/evaluation/historical-masked-2000-comparison.json), así que los productores y la evaluación comparten una única definición. El plan está en `training/campaign_plan.py`, la ejecución en `training/masked_campaign.py` y las predicciones trasladadas en `training/carried_predictions.py`. Nada de esto se ha ejecutado con datos reales.

### Variantes de presupuesto

La variante A reentrena desde cero cada ventana anual, con su ajuste, su validación y su selección. La variante B reentrena desde cero la primera ventana de cada ámbito y después cada 36 meses. Las ventanas intermedias no se ajustan y se predicen con el estado seleccionado en la última ventana reentrenada, que llamamos ancla. Con ese paso las anclas de US evalúan 2005, 2008, 2011, 2014, 2017, 2020 y 2023, y las de CN y US+CN evalúan 2011, 2014, 2017, 2020 y 2023, un subconjunto exacto de las de US. A diferencia de la alternativa B del [protocolo](walk-forward-2000.md#coste-y-alternativas-de-presupuesto), que solo evaluaba las ventanas reentrenadas, esta variante evalúa todos los años.

| Elemento de B | Regla |
| --- | --- |
| Estado usado | El estado seleccionado en el ancla para cada brazo y semilla: el ganador de la búsqueda con la semilla 42 y el finalista de cada otra semilla. No se cambian pesos, normalizadores ni selección |
| Información del ancla | Termina al final de su validación, el 1 de octubre del año anterior a su evaluación, con etiquetas maduras antes de esa fecha. El plan exige que esa fecha no sea posterior al comienzo de la calibración de cada ventana trasladada, y la predicción lo comprueba otra vez con los manifiestos |
| Filas y purga | Las de la vista de la ventana trasladada, con la purga por intervalo de etiqueta en sus propias fronteras. Son exactamente las filas que esa ventana tiene en A |
| Calibración | La comparación ajusta la calibración común con las predicciones del modelo trasladado en el tramo de calibración de esa ventana (los tres meses anteriores al año evaluado) y la congela antes de leer la evaluación |
| Ajuste y validación de la ventana trasladada | No se usan |

El precio de B es la antigüedad del modelo. La información del ancla termina 3 meses antes de su propia evaluación, 15 meses antes en el primer año trasladado y 27 en el segundo. B responde por tanto a otra pregunta, el error con reentrenamiento trienal, y no debe mezclarse con A. Como todos los brazos siguen el mismo calendario, la comparación emparejada dentro de cada variante conserva las mismas filas y la misma antigüedad.

En cada ventana reentrenada, cada brazo neuronal ajusta los dos candidatos del diseño (índices 0 y 10) con la semilla 42 y un finalista por cada semilla restante. Cada control de Titans-MAC hace lo mismo con los dos casos de búsqueda de su receta. Ridge ajusta sus tres alfas con la semilla 42, la única que le asigna la comparación, y XGBoost ajusta sus doce configuraciones con la semilla 42 y un finalista con cada una de las otras dos semillas. En cada ventana trasladada se hace una predicción por brazo y semilla.

| Variante | Ámbito | Ventanas reentrenadas | Ventanas trasladadas | Ajustes | Predicciones trasladadas |
| --- | --- | ---: | ---: | ---: | ---: |
| A | US | 19 | 0 | 1.007 | 0 |
| A | CN | 13 | 0 | 689 | 0 |
| A | US+CN | 13 | 0 | 689 | 0 |
| A | Total | 45 | 0 | 2.385 | 0 |
| B | US | 7 | 12 | 371 | 372 |
| B | CN | 5 | 8 | 265 | 248 |
| B | US+CN | 5 | 8 | 265 | 248 |
| B | Total | 17 | 28 | 901 | 868 |

Titans-MAC aporta 720 ajustes en A, y 272 ajustes y 336 traslados en B. Sin sus cuatro brazos quedarían los 1.665 ajustes de A y los 629 ajustes y 532 traslados de B.

Ajustes por brazo y semilla en cada ámbito:

| Brazo y semilla | A, US | A, CN o US+CN | B, US | B, CN o US+CN |
| --- | ---: | ---: | ---: | ---: |
| Cada referencia neuronal o control de Titans-MAC, semilla 42 | 38 | 26 | 14 | 10 |
| Cada referencia neuronal o control de Titans-MAC, semillas 43 y 44 | 19 | 13 | 7 | 5 |
| Ridge, semilla 42 | 57 | 39 | 21 | 15 |
| XGBoost, semilla 42 | 228 | 156 | 84 | 60 |
| XGBoost, semillas 43 y 44 | 19 | 13 | 7 | 5 |

En B cada brazo y semilla añade 12 predicciones trasladadas en US y 8 en CN y en US+CN. Cada configuración declara `max_training_jobs` y `max_prediction_jobs` iguales a su plan, de modo que añadir brazos, semillas o candidatos exige cambiar la configuración y su huella. Superarlos detiene la comprobación con el recuento previsto y el límite. La búsqueda temporal de referencias ya no tiene un tope fijo de 512 ejecuciones: un plan de versión 4 puede declarar `max_runs`, y sin ese campo conserva el límite anterior.

El 9 de octubre se eligió la variante A con todas las familias, registrada en [#363](https://github.com/GonxKZ/mars-titan/issues/363). Se prefirió entrenar cada ventana con todo el pasado disponible a reducir el coste, así que la medida de caudal ya no decide entre A y B. Con la [declaración ampliada](#declaración-preparada-de-las-familias-pendientes), A suma 4.680 ajustes en la campaña base, 3.915 en la etapa de adaptadores y 2.160 en la de políticas. La variante B conserva su configuración y sus pruebas, pero no es la elegida. Desde el 9 de octubre, además, la configuración declarada de B está retirada por decisión del autor: `campaign_plan.RETIRED` la lista y `launch_blockers` impide lanzarla, mientras `check` sigue contando sus trabajos y las copias de las pruebas ejercitan su código de traslado. La duración real se medirá con los primeros trabajos.

### Campaña A v2 con modelo conjunto

El 9 de octubre de 2026, antes de cualquier resultado, el autor tomó cuatro decisiones definitivas sobre la variante A, registradas en [#363](https://github.com/GonxKZ/mars-titan/issues/363). Son decisiones de diseño, no conclusiones. La [configuración v2](../../configs/baselines/historical-masked-campaign-a-v2.json) las declara en un archivo nuevo, y la de A y sus etapas no cambian.

1. Reentrenamiento anual desde cero en cada ventana. Se descarta el reentrenamiento cada 36 meses.
2. Todas las familias entrenan un único modelo conjunto US+CN con el [protocolo conjunto v3](walk-forward-2000.md#protocolo-conjunto-v3-de-la-campaña-a-v2), que empieza como US (primera validación en abril de 2004, 19 ventanas). China entra en el ajuste en cuanto tiene filas y sus métricas solo cuentan desde 2011, donde se cumple su historia mínima. Tres familias entrenan además modelos separados de US y CN como controles: `transformer_compact`, `titans_mac_online` y el brazo de MARS-TITAN `mars_titan_m1` (M1 con K = 1).
3. Los casos de búsqueda se ajustan con una sola semilla, la 42. El caso elegido por validación en cada ventana se repite con 43 y 44.
4. La parada temprana conjunta se conectará cuando exista su ejecutor, en preparación en otra tarea.

El control de MARS-TITAN es `mars_titan_m1` por cuatro motivos fijados antes de lanzar: es la variante principal (K = 1) de la declaración de ampliaciones, es la base de la familia de refinamientos K, su lector es el B de CM-v1 y su padre `titans_mac_online` también es un control. Padre y lector se entrenan así separados en el mismo ámbito, y el contraste no mezcla un padre conjunto con un lector separado. El ámbito de un mercado solo acepta un brazo con padre si ese padre también se ajusta en el ámbito. Solo los núcleos auxiliares de CM-v1 se incorporan sin declararlos.

| Elemento | Declaración en la campaña v2 |
| --- | --- |
| Ámbitos | US+CN con todos los brazos. US y CN solo con los tres controles separados |
| Brazos | Los de A más la GRU candidata, MARS-TITAN y CM-v1, copiados de la [declaración ampliada](#declaración-preparada-de-las-familias-pendientes) |
| `seed_policy` | Búsqueda con 42 y caso elegido con 43 y 44. Ridge y XGBoost son deterministas y tienen una sola semilla (XGBoost usa `subsample` y `colsample_bytree` iguales a 1,0, así que la semilla no interviene). La [configuración tabular v2](../../configs/baselines/tabular-historical-masked-v2.json) solo cambia los finalistas de XGBoost |
| `stopping` | `{"mode": "protocol"}`, presupuesto fijo del protocolo. Es el punto de conexión de la parada conjunta, que entrará como otro modo con sus grupos. Cualquier otro modo se rechaza hoy |
| `memory_options` | `accumulation_rows` de Titans-MAC, GRU candidata y CM-v1 y `recompute` de la GRU candidata, todos `pending`. `check` los lista como impedimentos y `run` se niega a empezar hasta que coincidan con el valor de su receta |
| `execution` | `{"order": "by_window"}`. La campaña recorre cada ventana completa antes de la siguiente |
| `numerics` | FP32 estricto: `float32_matmul_precision="highest"`, `cuda_matmul_allow_tf32=false` y `cudnn_allow_tf32=false`. Es el único valor admitido |
| `data_policy` | `real_edition_only`, el único valor admitido. Solo datos reales de la edición verificada, sin condiciones sintéticas, remuestreadas ni de aumento |
| `walk_forward_stages` | El [walk-forward por etapas](walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2) `staged_chain_v1`, único valor admitido: posentrenamiento desde la base k-1 con filas nuevas, predictor de la cadena con mejora estricta y RL con las tres evaluaciones anteriores a su validación (`fixed_prior_evaluations_v1`) |
| `online_controls` | El control `transformer_compact_online`, con padre `transformer_compact`, tope de escrituras del banco de `mars_titan_m1` y regla SGD con tasa, filas por paso, frecuencia y recorte pendientes |
| Límites | 2.322 ajustes, ninguna predicción trasladada y 153 trabajos del control en línea |

Cada finalista depende de todas las búsquedas de su brazo y ventana y, si el brazo parte de otro, del finalista del padre con la misma semilla. Su identidad guarda el caso elegido, la búsqueda ganadora y la huella de su recibo, de modo que un cambio en una búsqueda confirmada invalida el finalista al reanudar. La selección solo lee el MAE por sesión de validación del recibo, con desempate por identificador. La comparación resume cada semilla por separado y contrasta la media sesión a sesión de las semillas.

| Ajustes base | A ampliada (tres ámbitos) | A v2 |
| --- | ---: | ---: |
| Neuronales, US+CN | 87 × 13 = 1.131 | 87 × 19 = 1.653 |
| Neuronales, US | 87 × 19 = 1.653 | 12 × 19 = 228 |
| Neuronales, CN | 87 × 13 = 1.131 | 12 × 13 = 156 |
| Ridge y XGBoost | 17 × 45 = 765 | 15 × 19 = 285 |
| Total | 4.680 | 2.322 |

Por ventana conjunta hay 87 ajustes neuronales: 21 brazos o núcleos con dos casos y dos finalistas, y la GRU candidata con un caso y dos finalistas. Las etapas posteriores solo usan el modelo conjunto. Las configuraciones de etapa todavía declaradas son anteriores al walk-forward por etapas: 1.653 ajustes de adaptadores, 3.534 predicciones de la ablación de modalidades y 2.160 ajustes y 1.584 referencias de políticas. Con el diseño por etapas, la misma matriz de adaptadores no ajusta en la primera ventana y deja 1.566 ajustes con filas nuevas, 270 predicciones del padre congelado y 285 selecciones de la cadena (cinco brazos de referencia, tres semillas y 19 ventanas). Esos recuentos cambiarán al extender los adaptadores a todas las familias ([#446](https://github.com/GonxKZ/mars-titan/pull/446)) y la RL sobre la cadena ([#430](https://github.com/GonxKZ/mars-titan/pull/430)). La cinta US de políticas recorre 15 ventanas (2009 a 2023) y la CN 9 (2015 a 2023), las dos con las predicciones de la cadena del modelo conjunto en su mercado.

Las filas salen de los objetivos `targets-v3` con la regla de las vistas, en el [informe de recuentos](../../reports/data/campaign-a-v2-window-counts-20261009.json). Los de US y CN coinciden exactamente con las vistas ya verificadas. US+CN suma 125.553.766 filas de ajuste, 7.292.172 de validación, 3.721.369 de calibración y 15.434.391 de evaluación en sus 19 ventanas. Las vistas de la campaña v2 se prepararán sobre la edición v3.1, así que estos recuentos se [recalcularán](walk-forward-2000.md#comparación-con-los-controles-separados) con sus objetivos.

#### Política de datos

Por orden del autor del 9 de octubre, nada sintético en ninguna etapa, tampoco en el entorno de RL ni en el postentrenamiento: todo aprende con datos reales del dataset y con la división walk-forward de toda la serie. La campaña v2 lo declara con `data_policy="real_edition_only"` y `plan_campaign` lo comprueba antes de enumerar trabajos (`training/campaign_data_policy.py`). Exige que la campaña lea las vistas de la política histórica con máscaras y que ningún documento declarado por la campaña o por sus etapas registradas (configuración, configuración tabular, recetas, declaración de CM-v1, etapas de adaptadores, ablación y políticas, matriz de adaptadores y políticas comunes) nombre condiciones sintéticas, remuestreadas o de aumento, otra política de entradas o una fuente distinta de las vistas y las cintas reconstruidas de la edición. Los textos que explican una decisión pendiente no se examinan, sus claves sí. También exige que XGBoost no remuestree filas ni columnas. La comparación queda fuera porque no entrena: su remuestreo por bloques solo mide la incertidumbre. Las garantías durante la ejecución de las etapas de adaptadores y de políticas corresponden a sus módulos. Las pruebas con datos de juguete siguen permitidas porque no entrenan modelos de la campaña.

#### Precisión numérica

Por orden del autor del 9 de octubre (no inventar datos ni perder precisión), la campaña v2 declara FP32 estricto en `numerics`. Los 86 registros de precisión de la campaña de referencias del 6 de octubre (identidades y medidas de `real-campaign-20261006`) indican `float32_matmul_precision="highest"` y `cuda_matmul_allow_tf32=false`, pero `cudnn_allow_tf32=true`, el valor por defecto de PyTorch, que permite TF32 en los RNN de cuDNN (RNN, LSTM y GRU). La campaña v2 lo desactiva. Es un cambio de configuración declarado antes de lanzar, no una corrección por resultados, y no cambia ningún hiperparámetro (lote, épocas, paciencia ni `max_bin`).

El lanzador fija los tres indicadores en cada trabajo, antes de que su ejecutor cree modelos. Al confirmar exige que sigan igual y que el informe del ejecutor no registre otro valor con ninguno de los nombres que usan los informes del proyecto, guarda la precisión en cada recibo y rechaza al reanudar un recibo con otra. Las etapas de adaptadores y de ablación aplican la misma precisión de su campaña base, y la medida de caudal mide con ella. El caudal de 16.000 muestras-época por segundo de la proyección se midió con TF32 permitido en cuDNN, así que puede ser optimista para los brazos recurrentes.

#### Walk-forward por etapas y control en línea

El 9 de octubre, antes de cualquier resultado, el autor eligió el walk-forward por etapas con un control en línea ([#437](https://github.com/GonxKZ/mars-titan/issues/437)). El [protocolo](walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2) describe los roles de cada ventana, la regla del predictor de la cadena, el motivo por el que el reentreno de la ventana no es candidato, el diagrama de la cadena, el contrato de los recibos y el verificador de disjunción. Aquí se recoge lo que cambia en el plan.

- `walk_forward_stages` fija el diseño con un único valor admitido y exige `execution.order = "by_window"`. Con esa declaración, el calendario crea las selecciones de la cadena a partir de la etapa de adaptadores y exige dependencias concretas (`campaign_chain.check_staged`). Cada trabajo de posentrenamiento de la ventana k depende de los trabajos base que eligen su padre en k-1, y la primera ventana no tiene posentrenamiento. Cada trabajo de RL declara `predictor_seed` y depende de la selección de la cadena de todas las ventanas que lee. Las configuraciones de etapa anteriores al diseño se rechazan con su motivo hasta que adopten el contrato.
- `online_controls` declara `transformer_compact_online`. El plan crea un trabajo por ámbito, ventana y semilla (153 en total: 19 ventanas conjuntas, 19 de US y 13 de CN por tres semillas). Cada uno depende de los trabajos que eligen el estado de `transformer_compact` y de `mars_titan_m1` en la misma ventana y semilla. Sus trabajos llevan `regenerable=False`, porque sus predicciones salen de pasos en línea y la retención rodante no puede regenerarlas por inferencia. Mientras su regla tenga valores `pending`, `launch_blockers` impide lanzar la campaña. Su ejecutor y su brazo en la comparación se preparan en otra rama.
- La ablación de modalidades no incluye el control en línea, que no tiene un estado elegido que ablacionar. La estimación de disco lo excluye hasta tener el informe de su ejecutor, y la de horas lo acota con una predicción y, como máximo, un paso por fila de calibración y evaluación.

#### Orden ventana a ventana

Para liberar disco a medida que avanza, la campaña v2 se ejecuta ventana a ventana. `training/campaign_schedule.py` agrupa en una ventana de campaña las ventanas de todos los ámbitos con los cuatro tramos idénticos, con el nombre de la conjunta: de `fold-000` a `fold-005` están US+CN y US, y desde `fold-006` también CN. Cada ventana recorre estas fases, y después empieza la siguiente:

| Fase | Contenido |
| --- | --- |
| `base_search` | Casos de búsqueda de todos los brazos y ámbitos con la semilla 42 |
| `selection` | Caso elegido de cada brazo con el MAE de validación de sus recibos. No es un trabajo |
| `selected_case_seeds` | El caso elegido con 43 y 44 |
| `online` | El control en línea, desde el estado elegido de su padre y con el tope de su lector |
| `adapters` | Posentrenamiento desde la base de la ventana anterior con las filas nuevas |
| `chain` | Selección del predictor de la cadena de cada brazo base y semilla |
| `ablation`, `rl` | Ablación de modalidades y políticas, que leen la cadena de esta ventana y de las anteriores |
| `comparison` | Agregados por sesión de la comparación de la ventana en los tres ámbitos |
| `release` | Liberación de lo temporal de la ventana, a cargo de la retención rodante |

El plan comprueba que ninguna dependencia apunta a una fase posterior ni a una ventana posterior. La campaña base no tiene dependencias entre ventanas. Las únicas que cruzan ventanas son las del diseño por etapas: el posentrenamiento de k depende de la base en k-1 y la RL de k de las cadenas anteriores. `run`, `posttraining run`, `ablation run` y `rl run` aceptan `--window`, ejecutan solo los trabajos de esa ventana con sus dependencias y confirman solo la base de esa ventana. `schedule` muestra el orden sin leer datos. Admite `--adapter-stage`, `--ablation-stage` y `--rl-stage`, que hoy se rechazan porque sus configuraciones son anteriores al diseño por etapas. La persistencia de los agregados por ventana, que permitiría liberar las predicciones por fila antes del informe final, está pendiente y se coordinará con la nueva versión de la comparación.

```bash
uv run --no-sync python scripts/run_masked_campaign.py check \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json
uv run --no-sync python scripts/run_masked_campaign.py schedule \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json
uv run --no-sync python scripts/run_masked_campaign.py disjunction \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --views US+CN=<vistas conjuntas v3.1>/US+CN --views US=<vistas>/US --views CN=<vistas>/CN \
  --output <informe de disjunción>
uv run --no-sync python scripts/run_masked_campaign.py run \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --views US+CN=<vistas conjuntas v3>/US+CN --views US=<vistas>/US --views CN=<vistas>/CN \
  --output <campaña> --storage configs/baselines/historical-masked-campaign-storage.json \
  --window fold-000
```

#### Proyección de horas

`run_masked_campaign.py budget` aplica la fórmula de la [medición de caudal](#medición-de-caudal) a los trabajos exactos del plan: filas de ajuste por épocas entre el caudal, más las pasadas de validación, las predicciones finales y el calentamiento de las familias con memoria. El caudal puede ser uno común, con la inferencia a un múltiplo declarado, o el informe medido de `throughput`, y las épocas pueden sustituirse por las efectivas previstas de una parada temprana. Con 16.000 muestras-época por segundo, el caudal de la campaña de referencias del 6 y 7 de octubre, y la inferencia a 3 veces ese caudal:

| Etapa | A ampliada, 30 épocas | A v2, 30 épocas | A v2, unas 10 épocas efectivas |
| --- | ---: | ---: | ---: |
| Base neuronal | 11.373 h | 6.837 h | 2.319 h |
| Adaptadores | 1.884 h | 995 h | 995 h |
| Ablación de modalidades | | 27 h | 27 h |

Las 10 épocas efectivas son una estimación para la parada conjunta (rango de 8 a 14): la mediana de la mejor época en la campaña de referencias fue 3, la paciencia es 5 y el grupo para con su miembro más lento. Ridge, XGBoost y las políticas no escalan con el caudal neuronal y se estiman aparte entre 145 y 340 h sin medir. Con 4 semanas de reloj (672 h) y unas 580 h útiles, el factor de caudal necesario es horas neuronales / (580 − horas fijas): 10,1 con 250 h fijas y 7,0 con 100 h. Esta tabla conserva los adaptadores anteriores al walk-forward por etapas, que ajustaban con todo el tramo de ajuste de su ventana.

Con el walk-forward por etapas, los adaptadores solo recorren las filas nuevas de las ventanas con padre: 3.326.967 en US+CN, 2.775.272 en US y 447.458 en CN, frente a las 125.553.766 filas de ajuste del conjunto. El [informe de recuentos](../../reports/data/campaign-a-v2-window-counts-20261009.json) las recoge en `posttraining_rows`, contadas en los objetivos `targets-v3` con la regla de `campaign_chain.posttraining_rows`. Desde `fold-007`, las del conjunto menos las de US son exactamente las de la ventana CN con los mismos tramos. El caudal medido tras la nueva tubería de lectura, en `US+CN/fold-012` con FP32 estricto, es de 33.300 filas/s para un trabajo del Transformer compacto (40.400 para la GRU y 48.500 para DLinear), y tres trabajos a la vez con MPS suman entre 44.100 y 74.200 filas/s. Proceden del informe `reports/engineering/campaign-pipeline-20261009` de la rama `perf/campaign-pipeline`, todavía en revisión. Con 10 épocas efectivas e inferencia a 3 veces el caudal:

| Etapa | 33.300 filas/s, adaptadores anteriores | 33.300 filas/s, por etapas | 44.100 filas/s, por etapas |
| --- | ---: | ---: | ---: |
| Base neuronal y control en línea | 1.116 h | 1.116 h | 842 h |
| Adaptadores | 478 h | 30 h | 23 h |
| Ablación de modalidades | 13 h | 13 h | 10 h |
| Total neuronal | 1.607 h | 1.159 h | 875 h |

El control en línea suma 1,3 h a 33.300 filas/s, acotado con una predicción y como mucho un paso por fila de calibración y evaluación. Con el diseño por etapas y 33.300 filas/s, el factor de caudal necesario para 580 h útiles es 3,5 con 250 h fijas y 2,4 con 100 h. Con 44.100 filas/s baja a 2,7 y 1,8. Ninguna de las dos cifras cabe todavía en cuatro semanas. Las dos tasas son hipótesis de trabajo. La primera aplica el caudal de un Transformer solo a todas las familias, y Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata no se han medido, así que los brazos recurrentes y con memoria pueden ser más lentos. La segunda supone tres ranuras ocupadas todo el tiempo con el peor agregado medido. Las horas de la RL y de las referencias tabulares siguen fuera de la cuenta neuronal.

```bash
uv run --no-sync python scripts/run_masked_campaign.py budget \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --counts reports/data/campaign-a-v2-window-counts-20261009.json \
  --rate 33300 --epochs 10 \
  --stage configs/posttraining/historical-masked-adapter-stage-a-v2.json \
  --ablation-stage configs/evaluation/historical-masked-ablation-stage-a-v2.json \
  --fixed-hours 250 --target-hours 580
```

Si no cabe, en [#363](https://github.com/GonxKZ/mars-titan/issues/363) se ordenaron palancas de menor a mayor coste científico, todas con los datos completos y el reentrenamiento anual, sin elegir ninguna: el propio caudal, una parada más ajustada (12 épocas y paciencia 3), adaptadores con una semilla y 3 épocas, un solo caso de búsqueda, la continuación anual en caliente como comparación distinta, una prioridad de brazos declarada antes de lanzar y, en último lugar, renunciar a los controles separados.

### Parada y pérdida

Los casos neuronales se construyen con el diseño de referencias y la regla `stopping_rule(protocol)`: presupuesto fijo de 30 épocas con el mejor estado, MAE residual por sesión en validación, paciencia 5 como diagnóstico y mejora mínima 0,00001. La campaña rechaza protocolos con reglas distintas entre ámbitos, y cada caso debe coincidir con la regla antes de planificarse.

La pérdida común de los brazos neuronales es la pinball de [`quantile_head_v1`](../engineering/quantile-head.md), decidida en [#22](https://github.com/GonxKZ/mars-titan/issues/22). En la campaña sustituye a la familia de pérdidas del diseño de referencias, que alterna MAE, MSE y Huber según el índice. Los índices 0 y 10 conservan su tasa de aprendizaje, anchura, profundidad y abandono, y solo cambian la cabeza y la pérdida. El término de la mediana es la pérdida absoluta, coherente con la métrica principal. La búsqueda de la familia de pérdidas sigue en los estudios escalares y en el [control de la cabeza](../../configs/baselines/quantile-head-control-us.json). Si ese control obligara a volver a la salida escalar, haría falta otra configuración de campaña con otra huella.

Titans-MAC sigue la misma regla con su [receta de campaña](../engineering/titans-chronological-trainer.md#receta-de-la-campaña-y-casos-de-búsqueda), que activa la memoria con residual y LayerNorm junto a `gate_bias`, decidida en [#27](https://github.com/GonxKZ/mars-titan/issues/27). Para que la búsqueda sea equitativa, declara tantos casos como índices ajusta cada referencia neuronal, dos, con el mismo presupuesto fijo por caso y selección por validación, y la planificación rechaza otro número. La rejilla se deriva del diseño de referencias, sin datos. Los casos 0 y 10 usan la tasa 10⁻⁴ y se diferencian en anchura y dropout. En Titans la arquitectura queda fija por el emparejamiento de los cuatro controles, así que los dos casos varían la tasa: 10⁻⁴, la de las referencias, y 10⁻³, la de la receta v1. El recorte se mantiene en 1,0.

XGBoost no sigue la regla neuronal. Usa hasta 2.000 rondas con meseta de validación, un mínimo de 200 rondas, paciencia de 100 y la misma mejora mínima de 0,00001, sobre el mismo tramo de validación y con el MAE por sesión. Hay tres diferencias: la unidad es una ronda de boosting y no una época, se detiene en la meseta en lugar de agotar un presupuesto fijo, y el mejor modelo se sustituye con cualquier mejora estricta mientras la mejora mínima solo cuenta para la paciencia. No se alinea. El presupuesto fijo existe para que los controles neuronales emparejados apliquen el mismo número de actualizaciones, y XGBoost no forma parte de esos pares. Agotar siempre las 2.000 rondas, cada una con una lectura completa de páginas [estimada en 26 a 30 GB](../engineering/masked-tabular-comparators.md), multiplicaría el coste sin una comparación emparejada que lo justifique. Cambiar `BoostingSelection` cambiaría además la identidad de la ruta estricta. Ridge no tiene épocas y elige su alfa con el MAE por sesión de validación.

### Referencias tabulares

Ridge, HistGradientBoosting y XGBoost aceptan la retención reservada `heldout_full_train_sessions_v1`. Escriben validación, calibración y evaluación por fila con el mismo esquema que las referencias neuronales y resumen el ajuste sin tabla. Sin declararla conservan el recibo anterior de ajuste y validación.

El recorrido por ventanas no pasa por `posttraining/`. Tras [#386](https://github.com/GonxKZ/mars-titan/pull/386) la cola tabular de `posttraining/completion.py` lee la política, pero sigue esperando a una búsqueda neuronal completa sobre una sola vista, usa `run_tabular_search` y solo conserva ajuste y validación. Adaptarla habría exigido cambiar su contrato, así que la campaña llama directamente a los ejecutores tabulares.

HistGradientBoosting queda fuera del plan. Su ajuste concatena la matriz completa en memoria en `float64`, y con unos 15,4 millones de filas y 1.719 columnas serían unos 212 GB, frente a los 32 GB del equipo. Incluirlo exigiría muestrear filas, que rompe la población común, o un ajuste con memoria externa que aún no existe.

### Ejecución y recuperación

```bash
uv run --no-sync python scripts/run_masked_campaign.py check \
  --campaign configs/baselines/historical-masked-campaign-b.json
uv run --no-sync python scripts/run_masked_campaign.py prepare \
  --campaign <configuración> --parent <supervisión histórica> --output <vistas>
uv run --no-sync python scripts/run_masked_campaign.py run --campaign <configuración> \
  --views US=<vistas>/US --views CN=<vistas>/CN --views US+CN=<vistas>/US+CN --output <campaña> \
  --storage configs/baselines/historical-masked-campaign-storage.json
uv run --no-sync python scripts/run_masked_campaign.py sources --campaign <configuración> \
  --views US=<vistas>/US --output <campaña> --scope US --comparison <comparación>
```

`check` valida y cuenta sin leer datos. `prepare` crea las vistas de cada ámbito desde la supervisión histórica con los protocolos v2 y la recuperación de fronteras anuales (`recover_annual_boundaries`). Para un solo mercado escribe antes la proyección de la supervisión en ese mercado. Las vistas se comprueban frente a la comparación: misma política, mismas ventanas, mismos protocolos, una sola edición y todos los tramos con filas. Una vista de otro ámbito, de otro protocolo o de otra política se rechaza antes de crear la salida.

`run` llama a `require_learning_allowed` antes de abrir fuentes y otra vez antes de cada trabajo. Si la protección vuelve a estar vigente a mitad de campaña, `LearningHoldError` la detiene antes del siguiente trabajo y el resumen queda en `blocked`. La salida guarda la identidad de la campaña (configuración, comparación, configuración tabular, huellas de vistas y código) y un recibo por trabajo con su identidad, el intento, el informe del ejecutor, el MAE de validación y, para calibración y evaluación, la huella del Parquet, el número de filas y una huella de filas y objetivos independiente del orden. Cada predicción debe caer en su tramo, no contener filas de 2024 y tener tantas filas como la vista. Todos los trabajos de una ventana deben dar la misma huella de filas y objetivos, y el primero que difiera detiene la campaña.

`--storage` es obligatorio en `run`. La [declaración de almacenamiento](../../configs/baselines/historical-masked-campaign-storage.json) fija un margen de 8 GiB y los bytes medidos de tablas, estados, índices y cachés. Antes de crear la salida, `run` recorre los trabajos pendientes en el orden del plan y se niega a empezar si el pico proyectado más el margen supera el espacio libre. Antes de cada trabajo comprueba que su huella cabe sobre el margen y, si no, se detiene en `paused` sin empezarlo. Durante un trabajo, un espacio libre por debajo del margen activa la misma parada recuperable que una señal, en la siguiente barrera del ejecutor. Tras cada recibo libera los índices de observaciones y los estados de recuperación, como se describe en la [recuperación de checkpoints](../engineering/checkpoint-recovery.md#liberación-al-confirmar-en-la-campaña). El [presupuesto de disco](../../reports/engineering/campaign-storage-20261009/README.md) da las cifras y las opciones de retención que quedan por decidir.

Al reanudar, un trabajo con recibo e identidad iguales no se repite, después de comprobar las huellas de sus artefactos. Los trabajos sin recibo se rehacen: las referencias neuronales y XGBoost continúan su último intento desde su punto de control, y Ridge y las predicciones trasladadas empiezan un intento nuevo. Una salida de otra campaña, otras vistas u otro código se rechaza. Los trabajos CUDA se ejecutan de uno en uno bajo una única reserva de la GPU. Los trabajos CPU usan la concurrencia `cpu_workers` de la configuración tabular. Hoy Ridge y XGBoost se ejecutan en CUDA, así que esa concurrencia solo se aplicará cuando se conecte un ejecutor tabular de CPU.

Cuando una semilla de un brazo ya tiene su predictor elegido en una ventana (el ganador de la búsqueda cuando terminan todos los candidatos, el finalista o la predicción trasladada), `run` escribe el recibo de ventana que leen los entornos de refuerzo, uno por mercado, en `windows/<ámbito>/<ventana>/<brazo>/seed-<semilla>/<mercado>.json`. Se construye con el contrato de `environments/walk_forward_receipt.py` y se valida con su `read_window_receipt` antes de escribirlo:

| Campo | Valor en la campaña |
| --- | --- |
| `protocol`, `fold` | Protocolo v2 del mercado y ventana de la comparación declarada |
| `parent` | Trabajo que ajustó el estado elegido y huella de ese estado. En una ventana trasladada es el trabajo del ancla, y la campaña comprueba que la predicción trasladada partió de ese mismo estado |
| `labels_used_until` | Microsegundo anterior al inicio de la evaluación |
| `predictions` | Filas y `prediction_fingerprint` de la mediana emitida en calibración y evaluación, solo con las filas de ese mercado |

`labels_used_until` es una cota y no la maduración exacta de la última etiqueta. La calibración común usa el tramo anterior a la evaluación y la purga por intervalo de etiqueta obliga a que todas sus etiquetas maduren antes del final del tramo, así que ninguna etiqueta usada en ajuste, selección o calibración madura después. En una ventana trasladada el modelo dejó de aprender antes, pero su calibración también usa ese tramo. Al reanudar, un recibo de ventana ya escrito debe coincidir con el que se deriva de los trabajos confirmados. No hay recibos reales porque la campaña no se ha ejecutado.

`sources` publica el manifiesto de un ámbito para `evaluation.walk_forward_comparison`. Elige para cada brazo, semilla y ventana el ganador de la búsqueda, el finalista o la predicción trasladada, vuelve a exigir las mismas filas en todos ellos y valida el manifiesto con `load_sources` antes de publicarlo. Con la comparación declarada de 23 brazos falla y nombra los brazos sin productor. Para evaluar antes solo las referencias haría falta declarar, antes de ver resultados, una comparación con esos brazos.

### Puntos de extensión y etapas posteriores

| Familia | Brazos de la comparación | Tarea | Falta |
| --- | --- | --- | --- |
| GRU candidata | `gru_episodic` | [#383](https://github.com/GonxKZ/mars-titan/issues/383) | Declarar la sección `episodic_gru` en A y B tras medir memoria y caudal en `cuda:0`. La [entrada por ventana](../engineering/candidate-chronological-trainer.md#ventanas-walk-forward-de-la-campaña) ya existe |
| MARS-TITAN | `mars_titan_m0` a `mars_titan_m3`, `mars_titan_m1_k2`, `mars_titan_m1_k4` | [#366](https://github.com/GonxKZ/mars-titan/issues/366) | Declarar la sección `mars_titan` en A y B tras medir el lector en `cuda:0`. La [entrada por ventana](../engineering/mars-titan-extensions.md#ventana-walk-forward-y-traslado) ya existe, también para `mars_titan_m3` |
| CM-v1 | `cm_v1_b`, `cm_v1_bc`, `cm_v1_bm`, `cm_v1_bcm` | [#293](https://github.com/GonxKZ/mars-titan/issues/293) | Declarar la sección `cm_v1` en A y B tras medir la penalización C y el lector en `cuda:0`. La [entrada por ventana](../experiments/mars_titan_cm_v1/factorial.md#ventanas-traslado-y-campaña) ya existe |

Cada familia se conectará con su planificador y su ejecutor en el mismo registro. Hasta entonces el plan las informa como pendientes y no crea trabajos falsos. La GRU candidata ya está conectada: `masked_campaign.EXECUTORS` registra su ajuste por ventana y su traslado, y una sección opcional `episodic_gru` de la campaña declara su receta, la variante de cada brazo y la semilla de búsqueda. Sus predicciones contienen exactamente las filas de la vista y las de los demás brazos, comprobado con vistas del corpus técnico. Las configuraciones A y B no incluyen todavía esa sección, que forma parte de la declaración ampliada de A. Declararla en B añadiría 51 ajustes y 84 traslados (680 y 616 en total) y en A 135 ajustes (1.800). Antes de copiarla hay que fijar `accumulation_rows` o `recompute`, porque el tramo completo no cabe en 8 GB con el universo completo. La [medida en `cuda:0`](../../reports/engineering/cuda-checks-20261009/README.md#gru-candidata) da con 512 activos un pico neto de 1.157,8 MiB sin acumulación y de 256,6 MiB con `accumulation_rows=128`. Mientras tanto, la comprobación sigue informando del brazo como pendiente en [#383](https://github.com/GonxKZ/mars-titan/issues/383).

MARS-TITAN también está conectado con una sección opcional `mars_titan`, sin copiar todavía a las configuraciones A y B y presente en la declaración ampliada de A. La sección fija la receta del lector, la combinación de componentes de cada brazo, el brazo padre `titans_mac_online` y los brazos pendientes con su motivo. [M3](../engineering/m3-write-policy.md) ya tiene productor y estima sus escalas con el tramo de entrenamiento de cada ventana. Cada búsqueda depende de las búsquedas del padre en su ventana y cada finalista del finalista del padre con su semilla, y la campaña entrega ese padre al ejecutor. Declararla con sus seis brazos añadiría 1.080 ajustes en A y 408 ajustes con 504 traslados en B. Sus ejecutores declaran también `fastpath=False` mientras dura cada trabajo. La [orden de medición](#medición-de-caudal) ya recorre el lector de cada brazo, M3 incluido, aunque no se ha ejecutado. Su ajuste sin pasos coincide en CPU y `cuda:0` con M1 y M3. Los brazos conservan los nombres de la comparación declarada (`mars_titan_m0`, `mars_titan_m1`, `mars_titan_m2`, `mars_titan_m3`, `mars_titan_m1_k2` y `mars_titan_m1_k4`), que son los de sus recibos de ventana y los que tendría que nombrar la [etapa de políticas](#etapa-de-políticas-por-ventana) para usarlos como predictores.

CM-v1 está conectado con una sección opcional `cm_v1`, sin copiar todavía a las configuraciones A y B y presente en la declaración ampliada de A. La sección solo indica la [declaración del factorial](../../configs/titans/cm-v1-factorial.json) y la semilla de búsqueda. Los núcleos `cm_v1_core_b` y `cm_v1_core_c` son trabajos auxiliares con sus casos y finalistas en las ventanas reentrenadas, sin traslado propio ni recibo de ventana. B y B+M dependen del primero, y B+C y B+C+M del segundo, con la misma regla de búsquedas y finalistas que MARS-TITAN. El traslado de un brazo lleva consigo el núcleo y el lector elegidos. Declararla añadiría 1.080 ajustes en A, 360 de ellos de núcleos, y 408 ajustes con 336 traslados en B, 136 de ellos de núcleos. Sus ejecutores declaran `fastpath=False` mientras dura cada trabajo. La [orden de medición](#medición-de-caudal) recorre ya los dos núcleos, con la penalización C en `cm_v1_core_c`, y el lector de cada brazo. Los núcleos comparten la receta de Titans-MAC, también `accumulation_rows`. La penalización C [acumula por bloques de flujos](../experiments/mars_titan_cm_v1/factorial.md#acumulación-por-bloques-con-c) con el mismo gradiente que el tramo completo, así que el plan ya no rechaza la sección si esa receta fija la opción. Los brazos conservan los nombres de la comparación declarada (`cm_v1_b`, `cm_v1_bc`, `cm_v1_bm` y `cm_v1_bcm`), que son los que tendría que nombrar la [etapa de políticas](#etapa-de-políticas-por-ventana) para usarlos como predictores. Esa etapa rechaza los núcleos auxiliares, que no tienen recibo de ventana.

Titans-MAC sigue el mismo patrón y ya no figura como pendiente. La sección `titans_mac` declara la receta común de los cuatro controles, el control de cada brazo y la semilla de búsqueda, y `EXECUTORS` registra su [ajuste por ventana y su traslado](../engineering/titans-chronological-trainer.md#conexión-con-la-campaña-con-máscaras). Las configuraciones A y B la declaran, con los límites de trabajos recalculados, y siguen en estado `declared_not_executed`. `accumulation_rows` sigue en `null` en la receta. La [medida en `cuda:0`](../../reports/engineering/cuda-checks-20261009/README.md#titans-mac) estima para un tramo de `mac_online` con 5.023 flujos 14,74 GiB sin acumulación y 1,24 GiB con 128, y según el medidor ninguna variante cabe en 8 GiB sin acumular. El valor previsto es 128 y todavía no se ha fijado. Los dos ejecutores declaran `fastpath=False` mientras dura cada trabajo, que el recorrido cronológico exige y la orden de la campaña no fijaba.

El postentrenamiento con la [matriz de adaptadores de versión 2](../../configs/posttraining/adapter-matrix-v2.json) de [#364](https://github.com/GonxKZ/mars-titan/issues/364) es una etapa posterior que parte de los padres seleccionados en cada ventana. Su [ejecución por ventana](../engineering/masked-posttraining.md#etapa-por-ventana-de-la-campaña) está implementada en `posttraining/campaign_stage.py`, con una configuración por variante ([A](../../configs/posttraining/historical-masked-adapter-stage-a.json) y [B](../../configs/posttraining/historical-masked-adapter-stage-b.json)), y se ha comprobado sin pasos de optimizador. Prevé 3.915 ajustes en A y 1.479 ajustes y 2.436 traslados en B. En B los casos siguen el calendario de sus padres: se ajustan en las anclas y se trasladan sin ajuste a las ventanas intermedias. `LATER_STAGES` de `training/campaign_plan.py` la registra con la matriz de versión 2, la configuración de cada variante y su punto de entrada, sin tareas pendientes, y `run_masked_campaign.py posttraining check` o `run` la valida o la ejecuta sobre una campaña base confirmada. No se ha ejecutado.

La [etapa de políticas](#etapa-de-políticas-por-ventana) de [#137](https://github.com/GonxKZ/mars-titan/issues/137) es la segunda etapa posterior. `LATER_STAGES` la registra como `rl_policy_comparison`, con sus políticas comunes, la configuración de cada variante, su punto de entrada `simulation.campaign_stage:run_stage` y ninguna capacidad del motor pendiente desde [#422](https://github.com/GonxKZ/mars-titan/pull/422), porque las cuatro tienen sonda. `run_masked_campaign.py rl check` o `rl run` la valida o la ejecuta sobre una campaña base confirmada.

### Medición de caudal

Una sola orden mide en `cuda:0` las familias con entrenador y la etapa de adaptadores. Se preparó para elegir entre A y B en [#363](https://github.com/GonxKZ/mars-titan/issues/363). Con A ya elegida, su informe serviría para estimar horas y comparar opciones de memoria con su caudal. `--extensions` añade, solo para medir y estimar, las tres familias que A y B todavía no declaran, con la [declaración preparada](#declaración-preparada-de-las-familias-pendientes):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_EPISODIC_NATIVE=<enlace nativo> \
uv run --no-sync python scripts/run_masked_campaign.py throughput \
  --campaign configs/baselines/historical-masked-campaign-a.json \
  --campaign configs/baselines/historical-masked-campaign-b.json \
  --stage configs/posttraining/historical-masked-adapter-stage-a.json \
  --stage configs/posttraining/historical-masked-adapter-stage-b.json \
  --rl-stage configs/simulation/historical-masked-rl-stage-a.json \
  --rl-stage configs/simulation/historical-masked-rl-stage-b.json \
  --extensions configs/baselines/historical-masked-campaign-extensions.json \
  --views US=<vistas>/US --views CN=<vistas>/CN --views US+CN=<vistas>/US+CN \
  --first-view <vistas>/US/fold-000/manifest.json \
  --work <trabajo> --output <trabajo>/throughput.json
```

`--candidate-recipe` sigue disponible para añadir solo la GRU candidata con otra variante, y la orden rechaza combinarla con `--extensions`. La orden reserva la GPU con la regla de una sola carga y mide sobre la primera ventana:

| Familia | Recorrido medido | Opciones comparadas |
| --- | --- | --- |
| Referencias neuronales | Cada brazo y candidato: 50 lotes de ajuste con forward, pinball y backward tras 5 de calentamiento, y 50 de inferencia sobre la validación | |
| Titans-MAC | Cada control con la receta de campaña recorre `ChronologicalTrainer._train_pass` desde el inicio del tramo de ajuste, con `fastpath=False` como en `titans_fit`. Se cronometran 8 tramos entre barreras de paso tras 2 de calentamiento. Los dos casos de búsqueda comparten la medida, porque solo cambian la tasa de aprendizaje | `accumulation_rows` en `null` y 128 |
| GRU candidata | La receta y su variante principal `m1_k1` recorren `CandidateChronologicalTrainer._train_pass` con el módulo nativo | `accumulation_rows` en `null` y 128, con y sin `recompute` |
| MARS-TITAN | El lector de cada brazo conectado (M0, M1, M2, M3 y M1 con K = 2 y K = 4) recorre `ReadoutTrainer._train_pass` sobre un padre `mac_online` con pesos iniciales y congelado, porque los padres elegidos aún no existen y su coste no depende de sus pesos. Usa las fases y el calentamiento del padre, el banco empieza vacío y admite las etiquetas maduras con la escritura del brazo. Antes de recorrer M3 se estiman sus escalas con la regla de la campaña (`window_scalers`), sobre el tramo de entrenamiento de la ventana medida, y el informe guarda su huella y la duración de ese recorrido | Ninguna: la receta del lector no tiene alternativas de memoria |
| CM-v1 | Los dos núcleos recorren `ChronologicalTrainer._train_pass` con la receta del núcleo: `cm_v1_core_b` con C en `disabled` y `cm_v1_core_c` con la penalización, que en cada evento elige los flujos medidos y calcula su término de RᵀJR. Cada brazo recorre su lector M1 con K = 1 sobre el gemelo `disabled` de su núcleo, con retención reservoir en B y B+C y con centros fijos en B+M y B+C+M | Núcleos: `accumulation_rows` en `null` y 128, como Titans-MAC, porque comparten su receta y C también acumula por bloques. Lectores: solo la de su receta, que la estimación aplica a cada opción de los núcleos |
| Adaptadores | Cada caso de la matriz v2 con una semilla (brazos aplicables y continuación completa) sobre cada padre candidato con pesos iniciales, porque los padres elegidos aún no existen, con el lote de la matriz | |
| Políticas | El entorno financiero de cada mercado sobre una cinta sintética de 128 activos y 253 sesiones, con un ciclo fijo de las seis acciones, 2.048 transiciones tras 64 de calentamiento. La red `FinancialNetwork` con inferencia por lotes de 1 y de 16 observaciones, copias incluidas, y forward y backward de un minilote de 64 del objetivo PPO, sin optimizador | Contabilidad Python y nativa, donde el motor admite el mercado |

La inferencia de las familias cronológicas se mide con los parámetros congelados sobre 64 eventos del mismo tramo, tras 8 de calentamiento. Ningún recorrido crea un optimizador de PyTorch. Las familias cronológicas llegan hasta el paso con un optimizador propio de la medición, que solo cuenta los pasos pedidos y libera los gradientes, y al terminar cada opción se exige que los pesos no hayan cambiado, también los del padre congelado de cada lector. Un gancho global rechaza además cualquier paso durante toda la medición. Solo se registran filas por segundo, filas medidas, pasos pedidos sin actualización, memoria y contadores, nunca pérdidas ni errores. Una opción que no cabe en la memoria reservada queda como `out_of_memory` y la medición sigue con las demás.

Los contadores dicen qué cálculo recorrió la ventana medida. En `cm_v1_core_c`, `window_counters` guarda los grupos y flujos de C al empezar y al terminar la ventana. C mide un flujo cuando su número de observaciones es múltiplo de `frequency` (16), así que la orden exige antes de reservar la GPU que los tramos medidos sumen al menos 16 instantes. Con 8 tramos de 8 instantes se cumple y con `--segments 1` la orden se detiene. En cada lector, `window_counters` guarda los episodios admitidos, que deben superar la capacidad del banco (1.024) para que la medida incluya la retención y, en B+M y B+C+M, la selección con centros fijos. En M3 guarda también los admitidos, rechazados y expulsados del índice selectivo y los candidatos sin relevancia conocida.

Con esos caudales el informe estima las horas de GPU de cada variante por familia, opción, ámbito y brazo. Las referencias conservan su fórmula: 30 épocas de ajuste y validación más las predicciones finales. Titans-MAC, la GRU candidata, los lectores, los núcleos de CM-v1 y los casos de la matriz validan antes de la primera época y tras cada una, y vuelven a predecir la validación con el estado elegido. Un ajuste suma así sus épocas de ajuste, épocas + 2 pasadas de validación, calibración y evaluación, y un traslado solo calibración y evaluación. En Titans-MAC cada tramo predicho añade sus 12 meses de calentamiento, estimados con la densidad de filas del propio tramo, y lo mismo ocurre en los lectores de MARS-TITAN con la receta de su padre y en CM-v1 con la receta del núcleo. Las horas de un lector no incluyen el ajuste de su padre. Las de `titans_mac_online` se cuentan en Titans-MAC y las de los núcleos dentro de CM-v1, como trabajos auxiliares sin traslado, y el informe lo indica en `parents`. Cada padre de la matriz en una ventana reentrenada suma una pasada de ajuste y validación para su caché. Finalistas, traslados y padres sin elegir usan el candidato más lento. Para cada variante, `total_gpu_hours` suma las familias con la opción declarada en cada receta y con la opción más rápida que cabe en memoria, y `comparison` da A, B y el cociente B/A. Una familia sin estimación, como CM-v1 si un núcleo no cabe en memoria, queda en `without_estimate`. Ridge y XGBoost quedan como no medidos, porque medirlos ya sería ajustarlos. La estimación supone el mismo caudal en todas las ventanas y no incluye esperas de disco, índices, normalizadores ni reanudaciones. Las escalas de M3 son uno de esos normalizadores: cada ajuste M3 recorre una vez su tramo de entrenamiento, y su duración en la ventana medida queda en `write_scalers.seconds` sin sumarse a las horas.

La etapa de políticas tiene una estimación aparte, fuera de `total_gpu_hours`, porque mezcla entorno en CPU y red en GPU. Se desglosa por nivel, ámbito, brazo y predictor. Con `--extensions`, la etapa resuelve sus predictores en la campaña ampliada, 22 en vez de 11. Cada ajuste suma el presupuesto de transiciones con inferencia por lotes de 16, los 16.384 minilotes de sus recorridos, 17 validaciones completas sobre la cinta de validación y la evaluación de los tres costes. Un traslado evalúa los tres costes y una referencia lo hace sin red. Las sesiones de cada tramo se aproximan con días hábiles. La estimación es orientativa en los dos sentidos: el entorno nativo se mide a través de Python, que el motor C++ no necesita, y no se miden el paso de Adam, las oleadas de KLPO, el replay de Double DQN ni los puntos de control. Double DQN y KLPO se estiman con los minilotes de PPO. `tests/simulation/test_policy_throughput.py` (8 pruebas) fija la fórmula con caudales dados y recorre la medición en CPU sin crear optimizadores, con el gancho retirado al terminar y el rechazo de una medición que cambia pesos. Como referencia técnica, una pasada en CPU de `measure_stepping` con 128 activos y 512 transiciones, repetida tres veces con otros procesos en el equipo (carga media de 9,5), dio entre 241 y 388 transiciones por segundo con la contabilidad Python y entre 1.496 y 2.536 con la nativa en EE. UU. Cuando se midió, China solo admitía la contabilidad Python. El motor nativo ya aplica las reglas de acciones A, pero esa medida no se ha repetido. Con esa dispersión, la cifra no sirve para fijar el presupuesto. `native_policy_hours` sustituye estas aproximaciones con los tiempos del motor nativo medidos sin aprendizaje sobre cintas reconstruidas y cuenta el trabajo propio de cada motor, incluidos los 261.888 minilotes de cada ajuste de Double DQN. Con esas medidas, la etapa A necesita al menos 31,9 h en `cuda:0` con un trabajo cada vez y 9,6 h con cuatro procesos bajo MPS, sin contar los pasos de Adam ([informe](../../reports/engineering/rl-stage-native-20261009/README.md#proyección-de-la-etapa-a)).

El pico de memoria corresponde a la ventana medida. El informe registra para cada familia cronológica el máximo de observaciones por evento de esa ventana, y las ventanas posteriores tienen más activos. Antes de fijar una opción con el universo completo hay que repetir la orden con la ventana más poblada como `--first-view`. Las opciones elegidas cambian la identidad de cada receta y se registrarán en [#363](https://github.com/GonxKZ/mars-titan/issues/363) con la variante. La orden no se ha ejecutado.

#### Declaración preparada de las familias pendientes

[`historical-masked-campaign-extensions.json`](../../configs/baselines/historical-masked-campaign-extensions.json) prepara, sin activarlas, las secciones `episodic_gru` (variante `m1_k1`), `mars_titan` (seis brazos sobre `titans_mac_online`, M3 incluido) y `cm_v1`, junto a los límites que tendrían cada campaña y su etapa de políticas. Está en la carpeta de las campañas, así que sus rutas relativas significan lo mismo copiadas a cada archivo. `training/campaign_extensions.py` aplica las secciones a la campaña cargada con las reglas de `campaign_plan` (`extend_campaign`), y `run_masked_campaign.py extensions` comprueba sin leer datos que los límites preparados son los recuentos exactos:

| Variante | Campaña declarada | GRU candidata | MARS-TITAN | CM-v1 (núcleos) | Campaña ampliada | Adaptadores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, ajustes | 2.385 | 135 | 1.080 | 1.080 (360) | 4.680 | 3.915 |
| A, traslados | 0 | 0 | 0 | 0 | 0 | 0 |
| B, ajustes | 901 | 51 | 408 | 408 (136) | 1.768 | 1.479 |
| B, traslados | 868 | 84 | 504 | 336 | 1.792 | 2.436 |

La etapa de adaptadores no cambia, porque solo parte de las referencias neuronales. La de políticas resuelve sus predictores desde la campaña y pasa de 11 a 22:

| Etapa de políticas | Predictores | Ajustes | Traslados | Referencias | `max_training_jobs` | `max_evaluation_jobs` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A declarada | 11 | 1.368 | 0 | 792 | 1.368 | 792 |
| A ampliada | 22 | 2.160 | 0 | 1.584 | 2.160 | 1.584 |
| B declarada | 11 | 456 | 912 | 792 | 456 | 1.704 |
| B ampliada | 22 | 720 | 1.440 | 1.584 | 720 | 3.024 |

Activar la declaración consiste en copiar las tres secciones y los límites de cada variante a `historical-masked-campaign-{a,b}.json` y los límites de políticas a `historical-masked-rl-stage-{a,b}.json`. Las pruebas comprueban que ampliar una campaña cargada produce el mismo plan, los mismos recuentos y las mismas secciones que declararlas en su archivo. Los núcleos de CM-v1 comparten la receta de Titans-MAC. Si la medida confirma que `mac_online` necesita 128, B y B+C la conservan, porque la penalización C acumula por bloques con el mismo gradiente.

`tests/training/test_campaign_throughput.py` recorre en CPU el lector M0 sin enlace nativo y, con el enlace, M1, M2, M3, M1 con K = 2, los dos núcleos de CM-v1 con `accumulation_rows` en `null` y en 128 y sus cuatro lectores, siempre hasta el paso sin cambiar pesos y con el gancho activo en cada paso pedido. Comprueba que solo el núcleo con C recorre flujos medidos, que los bancos admiten episodios en la ventana, que M3 cuenta cada oferta madura como admitida o rechazada en su índice selectivo, que sus escalas coinciden con las de la regla de la campaña sobre el tramo de entrenamiento medido, que el padre congelado entra en la comprobación de pesos, las horas con los recuentos exactos, la medida de los lectores aplicada a cada opción de los núcleos, el calentamiento de la receta del núcleo, el núcleo sin memoria y la orden con la declaración preparada. `tests/training/test_campaign_extensions.py` fija los recuentos de las dos tablas, la equivalencia con la declaración en el archivo y el rechazo de límites que no coinciden, secciones repetidas o ajenas, otra carpeta y otra etapa de políticas. También comprueba que la receta del núcleo de CM-v1 admite `accumulation_rows=128` y sigue rechazando una receta que no cumple la regla común. La mutación dirigida aplicó de uno en uno 24 defectos a la lógica nueva (nombre de la opción de la receta, padre fuera de la comprobación de pesos, calentamiento de la receta del núcleo, CM-v1 fuera de la estimación, frontera de la comprobación previa de C, contadores invertidos o en el núcleo sin C, gemelo `disabled` del padre, recuento de políticas con los predictores antiguos, límites ignorados o solo acotados, secciones repetidas, carpeta y etapa de políticas sin comprobar, entre otros). En la primera pasada sobrevivieron dos, la medición de MARS-TITAN condicionada a CM-v1 y la carpeta sin comprobar con una sola variante absoluta. Se reforzaron sus pruebas y los 24 fallan ahora. Al admitir la acumulación con C se aplicaron cuatro defectos más (restaurar el rechazo del plan, tratar también los lectores como brazos con opciones, tomar la primera opción de cada brazo y medir los núcleos solo con su receta) y los cuatro fallan.

### Comprobaciones técnicas

Las pruebas de `tests/training/test_campaign_plan.py`, `test_masked_campaign.py`, `test_carried_predictions.py`, `test_campaign_throughput.py` y `test_tabular_retention.py` no ajustan modelos ni ejecutan pasos de optimizador y no usan la GPU. Comprueban los recuentos exactos de A y B por ámbito, brazo y semilla, que Titans-MAC declara tantos casos de búsqueda como las referencias neuronales, la causalidad de cada traslado, la regla de parada y la pinball de cada caso, el rechazo de límites superados, de reglas mezcladas y de configuraciones alteradas, los trabajos lanzados con su vista, ventana, semilla y caso, la reanudación tras una interrupción simulada, el bloqueo al empezar y a mitad de campaña, el rechazo de vistas o políticas mezcladas, filas distintas, objetivos distintos y filas de 2024, la concurrencia CPU y la cola acotadas, los recibos de ventana validados con `read_window_receipt` (padre elegido, cota de la última etiqueta y huellas por mercado), su comprobación al reanudar, el rechazo de un traslado que no parte del estado del ancla, el manifiesto aceptado por `walk_forward_comparison` en US y US+CN y la medición de caudal. Esta recorre en CPU referencias, adaptadores, Titans-MAC y GRU candidata con el corpus técnico sin cambiar pesos, con el gancho activo en cada paso pedido y fastpath restaurado, cronometra tramos y eventos entre barreras, registra una opción sin memoria y sigue, y estima las horas por familia, opción y variante con los recuentos exactos de trabajos de la campaña y de la etapa. Los ejecutores se sustituyen por dobles que escriben predicciones nulas con las filas exactas de cada vista, y las pruebas admiten la campaña con la protección temporal permitida de `learning_doubles`, como el resto de lanzadores con dobles. Las predicciones trasladadas se prueban en CPU con un ancla neuronal construida a mano con pesos iniciales y con una función tabular fija.

## Postentrenamiento

En las campañas A y B, cada postentrenamiento parte de un padre seleccionado en la misma ventana y se compara con ese padre congelado. En la campaña A v2 parte del estado elegido de la base en la ventana anterior y solo ajusta con las filas que ese padre no usó, según el [walk-forward por etapas](walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2). Los adaptadores se colocan solos y en combinaciones de uno, dos o tres puntos de inserción, con el mismo presupuesto de actualizaciones y la misma validación. Se ajustan solo con el tramo de ajuste de la ventana, se seleccionan con su validación y predicen calibración y evaluación con las mismas filas que la campaña base. Los objetivos ya derivados se describen en [adaptación predictiva](predictive-adaptation.md).

Las referencias neuronales de la campaña emiten cinco cuantiles. Sus adaptadores y su continuación completa optimizan la pinball media de esos niveles, la misma pérdida del padre, y la selección usa el MAE por sesión de la mediana. La corrección lineal residual queda excluida para estos padres con un motivo declarado: corrige un único valor escalar, la mediana guardada en la caché, y la corrección por nivel inicializada a cero ya es el brazo de la cabeza. Con padres escalares se conserva el diseño anterior, en el que la corrección residual se compara con la salida continua del padre.

## Refuerzo

Los entornos consumen únicamente predicciones fuera de muestra del walk-forward, identificadas por los [recibos de ventana](#ejecución-y-recuperación) de la campaña. La ejecución usa el precio posterior a la decisión, con costes y deslizamiento declarados y límites de posición. Un agente escrito a mano que intente leer información futura debe fallar o no obtener ventaja. Los resultados sobre entornos sintéticos no se presentan como resultados sobre FinMultiTime. El controlador KLPO terminal se describe en [su documento de ingeniería](../engineering/terminal-klpo-updates.md).

### Etapa de políticas por ventana

La etapa está declarada en las [políticas comunes](../../configs/simulation/historical-masked-rl-policies.json) y en una configuración por variante ([A](../../configs/simulation/historical-masked-rl-stage-a.json) y [B](../../configs/simulation/historical-masked-rl-stage-b.json)). `simulation/policy_plan.py` valida la declaración y enumera los trabajos sin leer datos, `simulation/window_tapes.py` monta las cintas de cada tramo y `simulation/campaign_stage.py` ejecuta, reanuda y confirma. No se ha ejecutado.

Cada política aprende y se elige con información anterior a su evaluación. La política de la ventana k se ajusta con los tramos de evaluación de las tres ventanas anteriores a k−1, se selecciona con el de k−1 y se evalúa en k. Se usan tramos de evaluación porque son los únicos con predicciones fuera de muestra de un ajuste que terminó antes, de modo que el predictor no vio ninguna etiqueta posterior a la primera decisión del tramo. La construcción lo exige con el `labels_used_until` de cada recibo. Las cuatro primeras ventanas del protocolo se reservan para el primer ajuste y la primera validación. EE. UU. tiene así 15 ventanas de política, que evalúan de 2009 a 2023, y China 9, de 2015 a 2023. 2024 sigue cerrado. La campaña A v2 conserva esas tres ventanas con la regla `fixed_prior_evaluations_v1`, y cada cinta lleva las predicciones del predictor de la cadena de su ventana. La ventana en expansión queda como sensibilidad declarada y desactivada, por el [sesgo de antigüedad](walk-forward-2000.md#rl-con-las-tres-evaluaciones-anteriores) que introduce en el universo de la política. El primer ancla y el número de ventanas de política no cambian.

El objetivo del trabajo exige medir el resultado financiero de todos los modelos, así que la etapa se declara en dos niveles:

| Nivel | Predictores | Brazos |
| --- | --- | --- |
| `all_predictors` | Todos los brazos con productor en la campaña base, con la semilla 42 | KLPO terminal y las tres referencias |
| `algorithms` | Transformer compacto y Titans-MAC en línea | Las tres variantes PPO y Double DQN |

El primer nivel no usa una lista fija. Se resuelve desde la configuración de la campaña, en su orden: hoy son rnn, lstm, gru, dlinear, el Transformer compacto, Ridge, XGBoost y las cuatro variantes de Titans-MAC. La GRU candidata entra en cuanto su sección `episodic_gru` se declare en la campaña, y lo mismo ocurrirá con MARS-TITAN ampliado y CM-v1 (B, B+C, B+M y B+C+M) cuando se registren sus productores, sin cambiar la etapa. Las referencias dependen de la predicción, porque el entorno invierte en el cuartil superior de puntuaciones positivas, así que también miden cada predictor sin aprendizaje. El segundo nivel compara algoritmos solo sobre la referencia y el núcleo que la propuesta contrasta, por dos motivos. Cada brazo aprendido multiplica los ajustes (con los cinco brazos sobre los 11 predictores, A necesitaría 3.960 ajustes en lugar de 1.368) y el contraste principal es KLPO, que ya cubre a todos los predictores. Para esos dos predictores la comparación de algoritmos usa los ajustes de KLPO y las referencias del primer nivel, sin repetirlos.

El universo de un ancla sigue la regla `median_traded_value_in_validation_v1`: activos admitidos en todas las cintas de ajuste y validación, con alguna predicción en validación, ordenados por la mediana de cierre por volumen de la validación y con desempate por identificador, hasta 128. La evaluación no interviene en esa elección. El universo es común a todos los predictores, porque la campaña base exige las mismas filas a todos sus brazos en cada ventana, y lo fija el primer predictor del nivel de algoritmos, el Transformer compacto. Si la cinta de evaluación excluye un activo del universo, por ejemplo por filas sin verificar, el trabajo registra sus episodios como fallidos con el motivo `universe_assets_excluded` en lugar de reducir el universo en silencio.

Un predictor sin filas del mercado en un tramo tampoco detiene la etapa. Si le faltan en el ajuste o la validación del ancla, sus ajustes y traslados no se ejecutan y registran todos sus episodios como fallidos con el motivo `predictor_without_predictions`, sin política ni selección. Si le faltan en la evaluación, los episodios de esa ventana fallan con el mismo motivo. Sus referencias solo necesitan la evaluación. El predictor del universo sí debe tener predicciones de ajuste y validación, o la etapa se detiene.

Los brazos comparten entorno, observaciones, las seis acciones y las semillas 42, 43 y 44:

| Brazo | Nivel | Motor | Objetivo |
| --- | --- | --- | --- |
| `klpo_terminal` (principal) | `all_predictors` | `native_klpo` | `klpo_terminal_token_full_v1` con el controlador `klpo_full_fresh_waves_v1` y dos actualizaciones confirmadas por referencia |
| `ppo_clip_full_kl` | `algorithms` | `native_ppo` | `ppo_clip_full_kl_v1` |
| `ppo_kl_penalty_adaptive` | `algorithms` | `native_ppo` | `ppo_kl_penalty_adaptive_v1` con KL objetivo 0,01 |
| `ppo_clip_kl_epoch_stop` | `algorithms` | `native_ppo` | `ppo_clip_kl_epoch_stop_v1` con KL objetivo 0,01 |
| `double_dqn` | `algorithms` | `native_ppo` | Double DQN |
| `cash`, `hold_initial`, `rebalance_50` | `all_predictors` | Contabilidad nativa | Sin aprendizaje: efectivo, compra inicial y regla fija del 50 % sobre la predicción |

El presupuesto se fija antes de evaluar: 262.144 transiciones por ajuste con 16 entornos, recorridos de 1.024 transiciones, cuatro épocas de minilotes de 64 y una validación cada 16.384 transiciones. La selección usa `ruin_count_then_mean_liquidated_log_growth` sobre la validación, con mejora mínima de 0,0001, paciencia 5 y sin parada temprana, así que todos los brazos y todos los predictores consumen el mismo presupuesto por ajuste. El MAE del predictor nunca interviene. Cada trabajo se evalúa con costes de 0, 10 y 25 pb. El capital es 1.000.000 en la moneda del mercado. Con los 10.000 de configuraciones anteriores, el cuartil superior de 128 activos recibía unos 78 por activo con la menor exposición, menos que un lote de 100 acciones A a cualquier precio por encima de 0,78 CNY y menos que una acción de muchas empresas estadounidenses.

| Variante | Nivel | Ajustes | Traslados | Referencias | Episodios de evaluación |
| --- | --- | --- | --- | --- | --- |
| A | `all_predictors` | 792 | 0 | 792 | 4.752 |
| A | `algorithms` | 576 | 0 | 0 | 1.728 |
| A | Total | 1.368 | 0 | 792 | 6.480 |
| B | `all_predictors` | 264 | 528 | 792 | 4.752 |
| B | `algorithms` | 192 | 384 | 0 | 1.728 |
| B | Total | 456 | 912 | 792 | 6.480 |

En B la política se ajusta en la primera ventana de política y cada tres, como la campaña base. EE. UU. ajusta en 2009, 2012, 2015, 2018 y 2021 y China en 2015, 2018 y 2021. Las ventanas intermedias evalúan sin ajuste la política elegida en su ancla, sobre el universo del ancla.

```bash
uv run --no-sync python scripts/run_masked_campaign.py rl check \
  --stage configs/simulation/historical-masked-rl-stage-b.json
uv run --no-sync python scripts/run_masked_campaign.py rl run \
  --stage configs/simulation/historical-masked-rl-stage-b.json \
  --views US=<vistas>/US --views CN=<vistas>/CN --views US+CN=<vistas>/US+CN \
  --campaign-output <campaña> --edition <edición desde 2000> --output <políticas>
```

`check` cuenta los trabajos y consulta las capacidades del motor sin leer datos. `run` llama a `require_learning_allowed` antes de nada y antes de cada trabajo pendiente, exige después todas las capacidades del plan y solo entonces abre fuentes y crea la salida. Lee de la campaña base confirmada el recibo de ventana y las predicciones del predictor elegido. El recibo debe pertenecer a la ventana y al mercado pedidos e identificar al padre elegido con la huella de sus predicciones. Los brazos y referencias de un predictor en una ventana reutilizan las mismas cintas, guardadas con su recibo y su universo, y una cinta cambiada se rechaza al reanudar. Un recibo sin predicciones del mercado solo se acepta si el trabajo elegido de la campaña tampoco las declara.

Cada trabajo confirma un recibo con sus cintas, su informe y un registro por coste, también cuando el episodio falla o se arruina. Un ajuste debe declarar el criterio de cartera sobre la validación, el presupuesto completo y la huella de la política elegida. Un traslado evalúa la política de su ancla sin transiciones ni selección y una referencia no aprende ni selecciona. Las métricas por predictor, brazo y coste cuentan episodios completos, arruinados y fallidos con sus motivos, y la media del crecimiento logarítmico liquidado se publica con su denominador. Un trabajo pausado se reanuda en su carpeta y los confirmados no se repiten.

`run` exige estas capacidades del motor antes de crear la salida. Cada una se comprueba con el motor instalado:

| Capacidad | Situación | Trabajos afectados |
| --- | --- | --- |
| `native_policy_reconstructed_tapes` | `mars-titan-ppo` con el esquema 4 ajusta PPO y Double DQN sobre cintas reconstruidas con su auditoría walk-forward, selecciona con el criterio de cartera en validación y evalúa el estado elegido sin aprendizaje. Se comprueba con `--capabilities` | PPO, KLPO y Double DQN |
| `native_klpo_financial_runner` | `mars-titan-klpo` recoge oleadas KLPO completas sobre las cintas, selecciona en validación y evalúa el actor elegido. Se comprueba con `--capabilities` | KLPO |
| `native_cn_a_share_rules` | El motor nativo aplica lotes, resto impar, bandas diarias y timbre de las acciones A. Se comprueba con una sonda sobre la biblioteca instalada | Todos los de China, también las referencias |

Con los binarios del preset `native-ppo-release` compilados, `rl check` declara disponibles las cuatro capacidades y conserva la huella de cada binario y su identidad de compilación. La recuperación de un ajuste real queda a cargo del ejecutor nativo, con sus puntos de control de cartera, órdenes, generadores, optimizador, replay y cursor. Los ejecutores, la regla de oleadas completas de KLPO y sus comprobaciones se describen en [políticas nativas sobre cintas reconstruidas](../engineering/native-policy-real-tapes.md).

Las pruebas no ajustan políticas ni ejecutan pasos de optimizador y no usan la GPU. `tests/simulation/test_policy_plan.py` (56 pruebas) fija los recuentos de A y B por nivel, que el primer nivel recorre todos los productores de la campaña e incorpora la GRU candidata al declararla, que todos sus ajustes comparten presupuesto y criterio de cartera, el orden causal de todos los trabajos, KLPO primero, las dependencias de cada traslado, el capital mínimo para un lote de acciones A y el rechazo de declaraciones alteradas, entre ellas una selección por MAE. `test_window_tapes.py` (15) comprueba con recibos y una edición sintética que un recibo de un ajuste posterior, de otro mercado, falsificado o renombrado como una ventana anterior se rechaza, que perturbar el precio o la predicción posteriores a una decisión no cambia ninguna observación ni acción hasta ese punto, que la exposición máxima solo opera activos del universo en aperturas ejecutables, que un activo excluido produce episodios fallidos, que solo la cinta de evaluación puede excluirlo y que un predictor sin predicciones de evaluación o de ajuste deja sus episodios fallidos con su motivo, sin política que evaluar en el segundo caso. `test_campaign_stage.py` (32) recorre A y B sobre una campaña base reducida con dos predictores, GRU y LSTM, con brazos aprendidos sustituidos por una política guionizada sin red, y comprueba cintas comunes por ventana, traslados con la política y el universo del ancla, un predictor sin predicciones de ajuste cuyos ajustes y traslados no se ejecutan, pausa y reanudación, el bloqueo al empezar y entre trabajos, la parada por capacidades ausentes antes de crear la salida, el rechazo de informes que ocultan episodios o rompen el contrato y el de una campaña base alterada. Con la biblioteca nativa compilada comprueba además que las referencias dan los mismos episodios con la contabilidad nativa y con la Python en esa campaña reducida. Otras pruebas comprueban que el bloqueo detiene la etapa antes de consultar o lanzar binarios, que las sondas leen la capacidad y la identidad que declara cada binario y rechazan uno sin la capacidad, y que un ajuste KLPO no declara más transiciones de las que caben en sus oleadas. `test_policy_recovery.py` (3) pausa un trabajo con el entrenador Python de PPO o Double DQN por debajo de su calentamiento y su recorrido, con momentos de Adam escritos a mano, y comprueba que el trabajo reanudado coincide con una ejecución continua en cartera, órdenes pendientes, cursor, generadores, optimizador, replay y recorrido, y en la pérdida y los gradientes de la actualización siguiente, calculados sin aplicarla. La cinta de esta prueba conserva precios y sesiones de la etapa con puntuaciones sintéticas, porque la campaña reducida emite pocas predicciones.

La mutación dirigida aplicó de uno en uno 26 defectos a la lógica nueva: orden de tramos, cota de etiquetas, ancla de los traslados, regla del universo, ventana y padre del recibo, contrato de los informes, episodios fallidos en el resumen, bloqueo entre trabajos, reglas chinas, capacidades antes de crear la salida, bandera de reanudación, KLPO primero y fórmula y pesos de la medición de caudal. En la primera pasada sobrevivieron cuatro (orden de tramos según los recibos, episodio completo sobre una cinta fallida, activo perdido en una cinta de ajuste y KLPO declarado en segundo lugar con controles coherentes). Se añadieron las pruebas que faltaban y los 26 fallan ahora. Los dos niveles y los predictores sin predicciones se sometieron a otros 19 mutantes (lista fija en el primer nivel, semilla o productor sin comprobar, algoritmos sobre todos los predictores, nivel de las referencias, recibo sin predicciones, predictor del universo, motivo del fallo, ejecutor lanzado sin datos de ajuste, referencias tratadas como ajustes, métricas mezcladas entre predictores y horas por nivel y predictor), y las pruebas los detectaron todos en la primera pasada.

## Métricas

La métrica principal es el MAE residual por sesión. Primero se promedian los activos de un mismo mercado e instante y después se aplica la ponderación temporal y entre mercados. Se registran también MSE y RMSE, acierto de dirección con convención de empates, correlación de rangos por sesión y, cuando la salida lo permita, pérdida pinball, cobertura y anchura de cuantiles con y sin calibración común. La diferencia frente a una referencia se informa como Delta_error = MAE_variante − MAE_base y como porcentaje 100·(MAE_base − MAE_variante)/MAE_base, con intervalos del 95 % por bloques temporales. El detalle está en [métricas](metrics.md).

## Cómputo

La campaña se ejecuta en una RTX 4070 Laptop de 8 GB con el perfil de energía de ahorro, que el equipo necesita para no apagarse por temperatura. En esas condiciones la GPU trabaja a unos 1.305 MHz con limitación térmica. La auditoría de preparación estima unas 85 h por familia, configuración y semilla si se reentrena cada ventana anual completa con 30 épocas. Es una hipótesis basada en caudales de ediciones anteriores. La variante A se eligió sin esperar a la [orden de medición](#medición-de-caudal), que no se ha ejecutado. La [proyección de la campaña A v2](#proyección-de-horas) da 3.341 h neuronales con unas 10 épocas efectivas a 16.000 muestras-época por segundo, frente a unas 580 h útiles disponibles. La duración real se medirá con los primeros trabajos, y el presupuesto por ajuste es el mismo para los brazos emparejados. Las optimizaciones de rendimiento y el presupuesto de disco de la campaña se están preparando en [#363](https://github.com/GonxKZ/mars-titan/issues/363).

## Decisiones pendientes antes de entrenar

- Terminar el código y las implementaciones pendientes y revisar un resumen de su estado. Hasta entonces la protección de aprendizaje sigue activa.
- Registrar la huella de la configuración que se lance. La campaña A v2 ya copia la declaración ampliada ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).
- Preparar y verificar las vistas de los tres ámbitos de la campaña v2 sobre la edición v3.1, cuando esa edición esté verificada, y recalcular con ella los recuentos y la proyección de horas.
- Fijar las opciones de memoria pendientes de la campaña A v2 y conectar la parada conjunta en su sección `stopping`.
- Fijar la regla del control `transformer_compact_online` (tasa, filas por paso, frecuencia y recorte) y conectar su ejecutor y su brazo en la comparación.
- Integrar las etapas de adaptadores y de políticas con el contrato de la cadena y ejecutar el verificador de disjunción sobre las vistas v3.1 y, después, sobre los recibos de cada ventana.
- Elegir, si la medida de caudal no cabe en el presupuesto, entre las palancas de la [proyección de horas](#proyección-de-horas).
- Comparación parcial con las referencias, si se quiere evaluarlas antes de conectar las demás familias.
- Revisar la configuración de evaluación declarada antes de ver resultados: familias de contrastes, base de los refinamientos K, mínimo de activos del Rank IC y longitud de bloque ([#32](https://github.com/GonxKZ/mars-titan/issues/32)). La [cabeza común](../engineering/quantile-head.md) y su calibración CQR ([#22](https://github.com/GonxKZ/mars-titan/issues/22)) están implementadas y el control de la cabeza sobre el Transformer compacto está declarado sin ejecutar.
- Semillas fijas y margen mínimo relevante de error, registrados antes de ver resultados.
- Política de retención de predicciones y checkpoints según el disco disponible, en preparación con el presupuesto de disco de la campaña.
- Compilar con LibTorch CUDA los ajustes nativos de la [etapa de políticas](#etapa-de-políticas-por-ventana), ya implementados y comprobados sin aprendizaje, y medir su caudal en `cuda:0` antes de fijar el presupuesto de transiciones.
- `accumulation_rows` de Titans-MAC, con 128 como valor previsto. Sin acumulación, ninguna variante cabe en la GPU con la población completa según la [medida en `cuda:0`](../engineering/titans-chronological-trainer.md#memoria-del-tramo-y-acumulación-por-bloques). Fijarlo cambia la huella de la receta de campaña y la identidad de sus trabajos y de los núcleos de CM-v1.
- Activar en A la [declaración preparada](#declaración-preparada-de-las-familias-pendientes) de la GRU candidata, MARS-TITAN y CM-v1 con las opciones de memoria fijadas. Si Titans-MAC fija `accumulation_rows`, los núcleos de CM-v1 lo heredan con la penalización C.
