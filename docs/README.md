# Documentación de MARS-TITAN

La documentación reúne el diseño de investigación, las decisiones de ingeniería y sus comprobaciones. La preparación de datos y los ensayos de coste ya tienen evidencia ejecutada. La arquitectura candidata y la comparación confirmatoria siguen pendientes.

| Para qué | Documento |
| --- | --- |
| Preparar datos y conocer su cobertura | [Preparación ejecutada](data/preparation.md), [esquema](data/schema.md) y [permisos](data/permissions.md). |
| Revisar la admisión de noticias | [Fechas, procedencia y problemas de contenido](data/news-policy.md), con auditorías separadas del piloto y del panel técnico. |
| Revisar precios y su universo | [Auditoría de los 5.023 CSV](../reports/data/price-audit.md), con exclusiones, acciones corporativas y límites de identidad. |
| Revisar los codificadores | [Pesos, tokenización y preentrenamiento](data/pretraining-audit.md), con repetibilidad comprobada en CUDA y límites históricos. |
| Consultar entrenamientos medidos | [Presupuesto experimental](../reports/resources/campaign-budget.md), [variantes de GRU y DLinear](../reports/baselines/reference-variants.md) y [RNN y LSTM](../reports/baselines/recurrent-comparison.md), con cuatro modalidades y macro. |
| Inspeccionar almacenamiento y memoria | [Bytes por capa y contrato Parquet](../reports/resources/storage-budget.md), con cobertura y límites de la conversión. |
| Consultar la preparación por bloques | [Límites, paridad y medición macro](../reports/data/streaming-validation.md), con recuperación por activo. |
| Revisar pruebas y límites de calidad | [Verificación local](../reports/resources/quality.md), con integración CUDA, cobertura, CRAP y mutación dirigida. |
| Entender la pregunta y cómo contrastarla | [Protocolo](research/protocol.md), [matriz de experimentos](research/experiment-matrix.md), [riesgos](research/risks.md). |
| Consultar la ampliación y su aportación candidata | [Alcance ampliado](research/research-expansion.md), [arquitectura candidata](research/candidate-architecture.md), [registro de hipótesis](research/novelty-ledger.md) y [crítica adversarial](research/adversarial-review.md). |
| Organizar el trabajo | [Objetivos y hitos](research/roadmap.md), [tablero de tareas](research/task-board.md) y [auditoría de redundancias y etiquetas](research/backlog-review.md). |
| Corregir la propuesta técnica inicial | [Revisión crítica del original](research/original-review.md). |
| Trabajar con los datos | [Ficha de FinMultiTime](data/finmultitime-card.md) y [contrato temporal](data/data-contract.md). |
| Estudiar variables macroeconómicas | [Catálogo de 140 candidatos](data/macro-catalog.md) y [mecanismos financieros](references/macro-review.md). |
| Consultar y actualizar datos complementarios | [Fuentes públicas y nueve archivos verificados](data/free-data-sources.md) y [capturas manuales con manifiesto](data/public-source-updates.md). |
| Diseñar y mantener la implementación futura | [Arquitectura](engineering/architecture.md), [entorno reproducible](engineering/reproducibility.md), [decisiones](engineering/decisions.md). |
| Ejecutar las tareas con herramientas concretas | [Guía de implementación y límites](engineering/implementation-guide.md) y [catálogo del tablero](research/task-board.md). |
| Consultar el avance de los entrenamientos | [Observatorio](https://gonxkz.github.io/mars-titan/), [contrato de publicación](engineering/observatory.md) y [frontend](../site/README.md). |
| Consultar las comprobaciones iniciales | [Verificación de la preparación](engineering/research-verification.md) y [registro de la base inicial](engineering/verification.md), con sus fechas y límites. |
| Consultar el estado del arte | [Bibliografía y biblioteca](references/README.md), [finanzas](references/finance-review.md), [memoria neural](references/neural-review.md), [libros](references/books-review.md). |
| Contrastar cerebro, recurrencia y eficiencia | [Aprendizaje y memoria](references/brain-review.md), [antecedentes recientes](references/frontier-review.md), [sistemas](references/systems-review.md) y [DeepSeek](references/deepseek-review.md). |
| Dimensionar y retomar el trabajo | [Dataset por bloques](engineering/full-dataset-training.md), [latencia](engineering/latency-budget.md), [plan de cómputo](engineering/compute-plan.md) y [checkpoints](engineering/checkpoint-recovery.md). |
| Evaluar ruido y fiabilidad | [Controles y límites](research/noise-and-reliability.md). |
| Revisar los enlaces de redes sociales | [Revisión inicial financiera](references/social-finance.md), [arquitectura y rendimiento](references/social-neural.md) y [ampliación de los ocho posts](references/social-followup.md). |
| Preparar evidencias y redacción | [Plantillas de informes](../reports/README.md) y [memoria de trabajo](../thesis/README.md). |

Cada cambio metodológico debe señalar qué decisión modifica y por qué. Las evidencias de ejecución tendrán un identificador y remitirán a datos, configuración y commit. Los archivos originales no se corrigen: sus revisiones se escriben por separado para conservar la trazabilidad.
