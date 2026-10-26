# Regularización y selección temporal

El núcleo permite comparar el decaimiento de pesos de AdamW con su referencia sin decaimiento. Las recetas nuevas son opcionales y no modifican la campaña histórica. Se cargan con `financial_run.load_recipe` y `titans_walk_forward.case_recipe`, y llegan a la fábrica de optimizadores que ya utilizan los entrenadores.

| Receta | `weight_decay` | Estado |
| --- | ---: | --- |
| [Histórica con máscaras](../../configs/titans/chronological-training-historical-masked.json) | 0 | Referencia existente |
| [Decaimiento 0,001](../../configs/titans/chronological-training-weight-decay-1e-3.json) | 0,001 | Preparada, sin ejecutar |
| [Decaimiento 0,01](../../configs/titans/chronological-training-weight-decay-1e-2.json) | 0,01 | Preparada, sin ejecutar |

El lector episódico de MARS-TITAN tiene otros dos perfiles, con [0,001](../../configs/titans/episodic-readout-weight-decay-1e-3.json) y [0,01](../../configs/titans/episodic-readout-weight-decay-1e-2.json), frente a su [receta histórica con valor cero](../../configs/titans/episodic-readout-historical-masked.json). Los consumen `mars_titan_run.load_recipe` y `mars_titan_run.case_recipe`. Conservan el padre congelado y solo regularizan los parámetros ajustables del lector. No cambian capacidad, admisión, recuperación ni políticas M0–M3.

Las tres conservan las cuatro variantes `transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`, el predictor de 64 unidades, la cabeza de cuantiles, K = 1, pinball, recorte de gradiente de norma 1 y FP32 estricto. Cada una declara los mismos dos casos de tasa de aprendizaje, 0,0001 y 0,001, y 30 épocas. El único cambio numérico entre recetas es `weight_decay`. Los seis pares de tasa y decaimiento tienen identidades de entrenamiento distintas.

En AdamW, el decaimiento multiplica los parámetros por `1 − learning_rate × weight_decay`, separado de la actualización adaptativa del gradiente. No equivale a añadir una penalización L2 a la pérdida de Adam. Esa distinción procede de Loshchilov y Hutter, *Decoupled Weight Decay Regularization*, [arXiv:1711.05101v3](https://arxiv.org/abs/1711.05101v3), revisado el 4 de enero de 2019. La implementación reutiliza [AdamW de PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html). Las intensidades de las recetas son decisiones de desarrollo, no valores óptimos demostrados.

El decaimiento se aplica a los parámetros que recibe el optimizador externo. No describe el olvido de los pesos rápidos por flujo, la consolidación episódica ni el decaimiento de calendario de un corrector. En continuaciones que deban conservar un padre existe además el [decaimiento anclado](adapter-variety.md), con identidad y comparación propias. Estas recetas usan AdamW sin ancla y no activan esa alternativa.

Las [referencias multimodales](configurable-reference-models.md) ya admiten dropout de 0, 0,1 y 0,2 en la fusión. El dropout recurrente interno y el del núcleo Titans permanecen a cero. La propuesta original elimina unidades durante entrenamiento y aproxima una combinación de redes al evaluar, como describe Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, [JMLR 15(56), 2014](https://jmlr.org/papers/v15/srivastava14a.html). En este proyecto se usa el [comportamiento de PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html), cuya capa es la identidad en evaluación. No se activa MC dropout ni se interpreta la dispersión de pasadas como incertidumbre calibrada.

## Comparación que queda por ejecutar

La selección conserva el mejor checkpoint según `session_mae` de validación temporal, con mejora mínima 0,00001. El presupuesto fijo mantiene las mismas actualizaciones en los controles. La paciencia de cinco evaluaciones registra una meseta, pero no detiene estos ajustes. Activar paradas independientes cambiaría el presupuesto emparejado y requiere un contraste distinto. Para una continuación, la selección puede conservar el padre si las actualizaciones posteriores empeoran. Guardar el estado para recuperar una interrupción tiene otra función que elegir el mejor modelo.

Cada comparación debe compartir filas, máscaras, cortes, calentamiento, semillas, pérdidas y presupuesto. La validación decide los hiperparámetros antes de abrir los tramos posteriores. El dropout de una referencia se compara conservando su anchura, profundidad y pérdida. Los índices actuales del diseño varían varios factores y no aíslan por sí solos su efecto. Los resultados se informan por sesión y con diferencias temporales emparejadas. El protocolo de la criba sigue separado del criterio que selecciona cada checkpoint.

Para atribuir los efectos, la comparación del decaimiento del lector debe usar el mismo checkpoint padre. La comparación del decaimiento del núcleo mantiene fija la receta posterior del lector. Combinar ambos es otro contraste, con su identificación y presupuesto registrados antes de evaluar. No se presenta el cambio conjunto como efecto exclusivo de uno de los componentes.

El decaimiento o el dropout pueden reducir información útil y empeorar el error. La parada o selección temporal también puede sobreajustarse a una validación reutilizada. No se ha medido una mejora de generalización con estas recetas. La protección de aprendizaje y el test final permanecen cerrados.

La regularización no verifica la veracidad de los datos ni elimina los errores del modelo. Se mantienen las ausencias explícitas, las fechas de disponibilidad, la exclusión de etiquetas futuras y el rechazo de valores no finitos. Las predicciones no se recortan para mejorar su error informado. Una menor pérdida de entrenamiento no acredita precisión sobre sesiones futuras.

## Comprobación técnica

Las [pruebas del núcleo](../../tests/training/test_regularization_profiles.py) y [del lector](../../tests/training/test_readout_regularization_profiles.py) cargan las recetas reales, comprueban sus identidades y los grupos de AdamW sin ejecutar pasos. También comprueban la ubicación del dropout de las referencias y que evaluar no consuma su RNG. Las pruebas de selección cubren presupuesto fijo, empate, mejora mínima, conservación del padre y validaciones no finitas.

La [verificación del 26 de octubre de 2026](../../reports/engineering/regularization-20261026/verification.json) registra 61 pruebas CPU distintas aprobadas entre ambas ejecuciones, incluidas las 21 nuevas. Seis integraciones antiguas de aprendizaje y CUDA siguen omitidas por la protección vigente. La referencia previa a las primeras mutaciones aprobó 22 casos, ya incluidos en los anteriores. Se detectaron cuatro [mutaciones dirigidas](../../reports/engineering/regularization-20261026/mutation-results.json): ignorar el decaimiento declarado en la fábrica, quitar el dropout configurado, seleccionar la última época en vez de la mejor y anular el decaimiento de una receta del lector. Cada mutación tuvo controles verdes antes y después, sin intentos de pasos de optimizador.

La [cobertura y el CRAP](../../reports/engineering/regularization-20261026/quality.json) se han calculado sobre los módulos reutilizados de selección y decaimiento anclado, con coverage.py 7.16.1 y Radon 6.0.1. El recorrido cubre 103/147 sentencias y 40/56 ramas, con CRAP máximo 90. CRAP utiliza cobertura de sentencias. El perfil sin ancla no ejercita todas las funciones del módulo anclado. Estos diagnósticos no acreditan la calidad de un entrenamiento completo ni de la ruta CUDA, y no hay código nuevo de producción en este cambio.
