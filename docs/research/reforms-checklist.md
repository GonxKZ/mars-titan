# Lista REFORMS del estudio

Estado: revisada el 29 de octubre de 2026, antes de entrenar ningún modelo de la campaña
desde 2000. La lista dice dónde documenta MARS-TITAN cada punto y qué falta. No sustituye la
revisión con el director ni acredita ningún resultado.

REFORMS (Kapoor y otros, 2024) reúne en ocho módulos 32 puntos que un estudio científico con
aprendizaje automático debería informar. Cada punto pide describir algo, no superar un
umbral. Aquí cada uno lleva su evidencia en el repositorio y uno de estos estados:

- **Documentado**: un documento o recibo versionado lo describe y, cuando procede, lo
  comprueba.
- **Parcial**: está descrito, pero falta una pieza que se nombra en la tabla.
- **Pendiente**: depende de trabajo o de resultados que todavía no existen.

La campaña no tiene resultados todavía. Los puntos que hablan de resultados describen qué se
informará y dónde. Los enunciados son un resumen propio de la tabla 1 del artículo, publicado
con licencia CC BY-NC 4.0 (`kapoor2024reforms` en la [bibliografía](../references/adversarial.bib)).

## Afirmación del estudio

El trabajo es una comparativa de soluciones. Pregunta si una memoria adaptativa de eventos,
la de Titans-MAC y las ampliaciones de MARS-TITAN, predice mejor el retorno residual de
acciones estadounidenses y chinas de FinMultiTime que modelos sin esa memoria, con las mismas
filas, ventanas y presupuestos ([protocolo](protocol.md#pregunta-y-alcance)). La métrica
principal es el MAE residual por sesión. Un efecto nulo o un coste desproporcionado también
responden a la pregunta.

## 1. Objetivos del estudio

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 1a | La población o distribución sobre la que se afirma algo | Las acciones del archivo retrospectivo de FinMultiTime de cada mercado, de 2000 a 2023 para el desarrollo y 2024 como test final ([universo](protocol.md#universo-y-selección-del-subconjunto), [ficha del conjunto](../data/finmultitime-card.md)). Los resultados quedan condicionados a la supervivencia del archivo ([precios sin ajustar](../data/unadjusted-prices.md)) | Documentado |
| 1b | Por qué se eligió esa población | La propuesta del trabajo fija FinMultiTime porque reúne precios, noticias, fundamentales y gráficos de dos mercados. La ficha y la [cobertura histórica](../data/historical-coverage.md) documentan qué parte de esa promesa se cumple en la copia | Documentado |
| 1c | Por qué se usa aprendizaje automático | Las soluciones comparadas son arquitecturas de aprendizaje. La comparación incluye el control cero y Ridge para medir si un modelo aprendido aporta algo sobre referencias triviales ([matriz de comparaciones](metrics.md#matriz-de-comparaciones-y-atribución-por-componentes)) | Documentado |

## 2. Reproducibilidad computacional

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 2a | El conjunto de datos, con un enlace o identificador persistente | El artículo de FinMultiTime (Xu y otros, arXiv 2506.05019v2) y los manifiestos con huellas SHA-256 de la [edición v3.1](../data/edition-v3-1.md) y su [recibo](../../reports/data/edition-v3-1-verification-20261011.json). La copia local no tiene una revisión de descarga acreditada y los datos no se redistribuyen ([procedencia](../data/permissions.md)) | Parcial: falta un identificador de la descarga original, que no se inventa |
| 2b | El código de entrenamiento y evaluación, versionado | Este repositorio, con licencia MIT. Cada recibo guarda el commit y las huellas de sus fuentes ([contrato de recibos](../engineering/campaign-receipts.md)) | Documentado |
| 2c | La infraestructura de cómputo | Portátil x86-64 con una RTX 4070 de 8 GB y 32 GB de RAM ([plataforma](../engineering/hardware-platforms.md)). Los recibos registran versiones, hilos y carga de la máquina | Documentado |
| 2d | Un README con instrucciones para obtener los resultados | El [README](../../README.md#empezar) y la [ejecución de la campaña](../engineering/campaign-execution.md) | Documentado |
| 2e | Una orden que reproduzca todos los resultados | La orden única que encadena criba, análisis, confirmación, posentrenamiento, refuerzo e informe está prevista en #794 (`scripts/run_study.py`) | Pendiente |

## 3. Calidad de los datos

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 3a | Fuentes, momento de la recogida y origen de la verdad de referencia | El [contrato de datos](../data/data-contract.md) exige `available_at <= prediction_at` por modalidad. Las noticias, los hechos contables y los macros tienen sus reglas de disponibilidad ([noticias](../data/news-policy.md), [admisión macro](../data/macro-admission.md)). La verdad de referencia no la anota nadie: es el retorno residual calculado con precios ([objetivo](target-definition.md)) | Documentado |
| 3b | El marco de muestreo | El archivo retrospectivo distribuido, no una lista histórica de constituyentes, con dos cohortes declaradas ([cohortes](../data/cohort-policies.md)) | Documentado |
| 3c | Por qué el conjunto sirve para la tarea | La [cobertura por año](../data/historical-coverage.md#recuento-anual) muestra qué modalidades existen en cada periodo, y la edición desde 2000 conserva sus ausencias con máscaras en lugar de recortar filas ([máscaras](../data/historical-input-masks.md)) | Documentado |
| 3d | La variable respuesta y sus estadísticos descriptivos | La definición, su reloj y un ejemplo reproducible están en el [protocolo](protocol.md#reloj-de-decisión-y-objetivo) y en el [ejemplo del objetivo](target-definition.md) | Parcial: faltan los estadísticos descriptivos del objetivo en el periodo de desarrollo, sin 2024 |
| 3e | El tamaño de la muestra y la frecuencia de la respuesta | 18.698.976 muestras, 5.023 activos codificados y 5.008 con etiquetas aceptadas, con recuentos por mercado y motivos de exclusión ([recibo de la edición](../../reports/data/edition-v3-1-verification-20261011.json)) | Parcial: falta la frecuencia del signo del objetivo, que acompañará a los estadísticos de 3d |
| 3f | El porcentaje de datos ausentes | Recuentos por año y modalidad en la [cobertura histórica](../data/historical-coverage.md#recuento-anual) y recuentos de sustitución y arrastre de texto en el recibo de la edición | Parcial: faltan los porcentajes de ausencia por modalidad y mercado en una sola tabla |
| 3g | Por qué los datos de evaluación representan la población | Las ventanas walk-forward recorren todos los años elegibles con las mismas filas para todos los modelos, y 2024 queda cerrado hasta la [apertura única](../engineering/final-test-opening.md). La [auditoría de población](population-and-results-audit.md) recalcula recuentos y orden temporal | Documentado |

## 4. Preprocesamiento

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 4a | Las muestras excluidas y por qué | Los motivos de exclusión de etiquetas del recibo de la edición, la [auditoría de precios](../../reports/data/price-audit.md) y las reglas de admisión de las cohortes | Documentado |
| 4b | Cómo se tratan muestras imposibles o corruptas | Las filas con desorden por redondeo de la [edición v3.1](../data/edition-v3-1.md#filas-con-desorden-por-redondeo), los [huecos de precios](../data/historical-price-gaps.md) y la auditoría de precios. Nada se rellena con valores inventados | Documentado |
| 4c | Todas las transformaciones de los datos brutos a la entrada del modelo | La [preparación multimodal](../data/preparation.md), la [materialización histórica](../data/historical-materialization.md), el [esquema](../data/schema.md) y la conversión exacta de la [edición v3.1](../data/edition-v3-1.md#conversión-exacta-del-texto-de-la-fuente) | Parcial: falta un diagrama único del archivo bruto a la entrada del modelo |

## 5. Modelado

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 5a | Todos los modelos entrenados | [Titans-MAC](titans-mac-architecture.md), el [Transformer compacto](../engineering/compact-transformer-reference.md), la [referencia GRU con banco episódico](candidate-architecture.md), las [referencias configurables](../engineering/configurable-reference-models.md), [LightGBM](../engineering/external-boosting.md) y las variantes de `docs/experiments/` | Documentado |
| 5b | Por qué esos tipos de modelo | El [protocolo](protocol.md#pregunta-y-alcance), la [revisión neuronal](../references/neural-review.md) y la correspondencia entre ecuaciones, módulos y pruebas de Titans-MAC | Documentado |
| 5c | El método de evaluación, con sus particiones | El [walk-forward v2](walk-forward-2000.md), con la purga por el intervalo de cada etiqueta, y los [cortes de la validación temporal](evaluation-protocol.md#cortes-y-purga) | Documentado |
| 5d | Cómo se eligieron los modelos que se informan | La validación elige con el MAE residual ([particiones](protocol.md#particiones-ajuste-y-calibración)). La [búsqueda](../engineering/reference-search.md) registra cada intento y la [declaración del efecto de selección](../../configs/evaluation/historical-masked-2000-selection-effect.json) mide cuánto cuesta elegir | Documentado |
| 5e | El ajuste de hiperparámetros | Doce configuraciones por familia en la [búsqueda](../engineering/reference-search.md#búsqueda-finalistas-y-controles) e hiperparámetros fijos en la [criba de variantes](../experiments/variant-screening-v1.md) | Parcial: falta la rejilla pequeña de la confirmación, igual para variante y control (#794) |
| 5f | Que las comparaciones usan referencias adecuadas | La [matriz de comparaciones](metrics.md#matriz-de-comparaciones-y-atribución-por-componentes), con el control cero, Ridge, LightGBM, el Transformer compacto y los controles con las mismas entradas | Documentado |

## 6. Fugas de información

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 6a | Que el preprocesamiento y el ajuste solo usan datos de entrenamiento | El residualizador usa pares ya conocidos en t, el calibrador se congela antes de evaluar y cada normalizador se registra con su corte ([protocolo](protocol.md#modalidades-y-disponibilidad)). Las [vistas de información](../engineering/information-views.md) y el [escáner de fugas](integrity-bench.md#escáner-de-fugas-en-las-entradas) lo comprueban | Documentado |
| 6b | Cómo se tratan dependencias y duplicados entre entrenamiento y prueba | La [purga por intervalo](walk-forward-2000.md#purga-por-el-intervalo-de-cada-etiqueta) separa tramos. La incertidumbre remuestrea bloques de sesiones que conservan juntos los activos, y las semillas no cuentan como mercados independientes ([métricas](protocol.md#métricas-e-interpretación)) | Documentado |
| 6c | Que cada variable es legítima y no filtra la respuesta | La regla de disponibilidad de cada modalidad, las versiones de publicación de los hechos contables y las vendimias de los macros. El protocolo declara que un codificador preentrenado publicado después del periodo puede aportar conocimiento retrospectivo | Documentado |

## 7. Métricas e incertidumbre

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 7a | Las métricas y por qué se elige con una de ellas | El MAE residual por sesión como principal, con Rank IC, dirección, cuantiles, intervalos y cartera como secundarias ([métricas](metrics.md), [protocolo](protocol.md#métricas-e-interpretación)) | Documentado |
| 7b | La incertidumbre y cómo se calcula | Bootstrap por bloques de sesiones sobre diferencias pareadas y la regla de dispersión entre semillas ([semillas](protocol.md#dispersión-entre-semillas-en-los-pilotos-de-memoria)) | Documentado, sin resultados todavía |
| 7c | Las pruebas estadísticas y sus supuestos | Diebold y Mariano con la corrección de Harvey y Holm dentro de cada familia, y StepM de Romano y Wolf ([métricas](metrics.md)). El [PBO y el DSR](backtest-overfitting.md) acompañan a la cartera como diagnósticos | Documentado |

## 8. Generalización y límites

| Punto | Qué pide | Dónde está | Estado |
|---|---|---|---|
| 8a | La evidencia de validez externa | Dos mercados con monedas y reglas distintas, por separado y en conjunto, y un año final que no interviene en ninguna elección. No hay un conjunto externo al archivo | Pendiente de los resultados |
| 8b | Dónde no se espera que valgan las conclusiones | El sesgo de supervivencia, los costes ilustrativos sin préstamo ni impacto ([cartera](long-short-portfolio.md#qué-no-permite-afirmar)), el conocimiento retrospectivo de los codificadores preentrenados y la ausencia de operaciones reales | Documentado |

## Huecos

| Hueco | Qué falta | Dónde se resolverá |
|---|---|---|
| 2e | La orden única que reproduce todos los resultados | #794 |
| 3d y 3e | Estadísticos descriptivos del objetivo residual y frecuencia de su signo en el periodo de desarrollo, sin leer 2024 | Pendiente de una tarea propia |
| 3f | Porcentajes de ausencia por modalidad y mercado en una tabla | Pendiente de una tarea propia |
| 4c | Un diagrama único del archivo bruto a la entrada del modelo | Pendiente de una tarea propia |
| 5e | La rejilla de la confirmación | #794 |
| 8a | Los resultados en los dos mercados y el test final | La campaña, bloqueada hasta su aprobación |

## Referencia

Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman,
J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D.,
Roberts, M., Salganik, M. J., Serra-Garcia, M., Stewart, B. M., Vandewiele, G. y Narayanan, A.
(2024). REFORMS: Consensus-based recommendations for machine-learning-based science.
*Science Advances, 10*(18), eadk3452. https://doi.org/10.1126/sciadv.adk3452
