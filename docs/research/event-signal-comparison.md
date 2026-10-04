# Eventos públicos y comparación de entradas completas y reducidas

Fecha de corte: 4 de octubre de 2026. Diseño de una línea ampliada de MARS-TITAN, asociado a O3, O4 y O6. Se conserva el [candidato original](candidate-architecture.md). No se implementa ni entrena una arquitectura nueva ni se modifica la campaña activa.

Se plantean dos preguntas separadas: si aprender a anticipar eventos públicos ayuda a predecir retornos residuales y qué información necesita cada modelo para conservar calidad con menor coste. El estudio incluye acontecimientos empresariales y macroeconómicos, políticos y sociales, tanto por separado como mediante una representación compartida. La repetición y la atención se estudian en la [revisión complementaria](attention-replay-review.md).

## Referencia completa y variantes reducidas

La referencia principal mantiene precios, texto, fundamentales, gráficos y los 140 indicadores macro admitidos. Las variantes de entradas reducidas son nuevos contrastes identificados. Usan exactamente la cohorte admitida por la referencia completa, con las mismas fechas, etiquetas, exclusiones y cortes. No ganan filas por exigir menos modalidades. Estudiar después su cobertura en una población mayor sería otra comparación.

Esta ampliación permite investigar retirada de variables o modalidades sin sustituir la línea principal. Las reglas anteriores que mantienen todas las entradas siguen describiendo la referencia completa y sus ablaciones internas de memoria. No se reinterpretan resultados anteriores como si se hubieran entrenado con las variantes reducidas.

| Contraste | Qué información conserva | Qué demostraría una comparación favorable |
| --- | --- | --- |
| Representación comprimida | Todas las fuentes necesarias para construirla | Menos estado o cálculo después de codificar, incluyendo el coste de compresión |
| Menos transformaciones derivadas | Las fuentes originales y toda la historia necesaria | Que el modelo puede prescindir de esas transformaciones explícitas, no de sus fuentes |
| Subconjunto de variables | Solo las variables declaradas y sus dependencias | Calidad con menos adquisición o procesamiento, si se elimina su uso en todo el recorrido |
| Modalidades reducidas | Las modalidades conservadas, con entrenamiento adaptado | Calidad del modelo reducido sobre la misma población, no mayor cobertura por relajar admisión |
| Estudiante destilado | Entradas reducidas en uso, información del profesor durante aprendizaje | Transferencia con menor coste de inferencia, contabilizando profesor y objetivos generados |

El catálogo contiene 70 entradas `raw` y 70 `derived`. Una tasa interanual requiere observaciones anteriores, unidades y revisiones compatibles. Los últimos 70 valores originales no reconstruyen automáticamente las 140 entradas. Se contarán retardos, observaciones y bytes. Una imagen construida determinísticamente a partir del mismo prefijo de precios tampoco añade una observación externa, aunque su representación y codificador puedan facilitar el aprendizaje.

Cada variante tendrá una vista de entrada declarada que se aplica antes de codificar, recuperar, filtrar estados o calcular confianza. Si se retira una señal, tampoco podrá llegar indirectamente por el HMM, router, índices, memoria, calibrador o puerta de cálculo. Un selector que primero lee todas las señales sigue pagando ese acceso. El preentrenamiento y la destilación se registran por separado de la información disponible durante inferencia.

## Qué significa predecir un evento humano

El objeto son eventos públicos definidos mediante registros verificables. No se atribuye acceso a intenciones o pensamientos. Se distinguen tres objetivos:

1. **Ocurrencia.** Probabilidad de que una clase de evento ocurra dentro de un horizonte.
2. **Contenido.** Valor, categoría o magnitud que se anunciará, cuando exista una etiqueta definida.
3. **Resultado financiero posterior.** Retorno residual o distribución de retornos tras el corte de decisión.

Una publicación programada puede tener un calendario conocido. Predecir su fecha no demuestra anticipar su contenido. Una sorpresa respecto al consenso requiere un consenso archivado y disponible antes del anuncio. Si no existe, se puede estudiar una diferencia respecto a otra referencia explícita, pero no llamarla sorpresa del mercado. Predecir que habrá una noticia tampoco equivale a predecir la reacción de su precio.

La línea empresarial y macroeconómica puede estudiar resultados corporativos, cambios de previsiones, anuncios y decisiones de política económica. La línea política y social puede estudiar categorías públicas de decisiones regulatorias, relaciones internacionales o acontecimientos sociales con procedencia acreditada. El vocabulario y los criterios de inclusión se fijan antes del desenlace. No se construye la pregunta retrospectivamente desde el texto de una noticia futura.

La comparación comienza con horizontes ligados a sesiones y definidos en desarrollo. Cada objetivo tiene su propia fecha de maduración. Una etiqueta negativa de ocurrencia solo se confirma cuando termina el horizonte y existe cobertura suficiente para observar el evento. La ausencia de una noticia en un proveedor no se interpreta automáticamente como ausencia del acontecimiento.

El registro necesita identidad del evento, tipo, fuente, fecha de ocurrencia, primera publicación, disponibilidad acreditada, revisiones y cobertura del intervalo. Las menciones duplicadas se agrupan y la regla de agrupación utiliza únicamente lo conocido hasta cada corte. Las correcciones se incorporan como revisiones nuevas. No se reescriben las predicciones históricas para hacerlas coincidir con una versión posterior del registro.

Esta taxonomía y su cobertura aún no están auditadas sobre el corpus del proyecto. Las cabezas de eventos son un diseño condicionado a esa admisión. No se presentan como un dataset ya construido ni como una capacidad entrenada.

## Antecedentes y diferencias de tarea

[StockMem](https://arxiv.org/abs/2512.02720v1) estructura noticias y experiencias financieras con resultados observados. El estudio leído usa cuatro empresas y excluye retornos entre −1 % y 1 %. Motiva una recuperación estructurada, pero no valida el panel completo ni la ocurrencia futura de eventos. Sus explicaciones llamadas causales no identifican efectos económicos.

[Daily Oracle](https://proceedings.mlr.press/v267/dai25l.html), ICML 2025, distingue cortes de noticias y preentrenamiento al evaluar preguntas futuras. La revisión humana descrita cubre 60 preguntas. Para el proyecto, la definición prospectiva de preguntas y clases será un requisito propio, sin atribuir a todo el benchmark una auditoría manual que no tiene.

[The Power of Simplicity in LLM-Based Event Forecasting](https://aclanthology.org/2025.realm-1.32/), taller REALM 2025, compara estadísticas estructuradas, titulares y razonamiento ReAct sobre MIRAI. Es un control pertinente para preguntar cuánto aporta el contexto adicional. No demuestra que las noticias sean prescindibles para retornos residuales ni que más pasos internos siempre ayuden.

[Hawkes Attention](https://proceedings.mlr.press/v300/tan26a.html), AISTATS 2026, modela tipo y tiempo mediante historia de eventos. Sus tareas principales son generales, no financieras. Una intensidad condicionada a la historia puede ser una alternativa a probabilidades por horizonte, pero añade coste de atención e integración. Se estudiaría después de tasas históricas, calendarios y un modelo discreto sencillo.

[TimeFilter](https://proceedings.mlr.press/v267/hu25ac.html), ICML 2025, filtra dependencias entre parches y variables. Sigue necesitando las variables con las que construye el grafo. Restringir relaciones después de recibir las entradas no acredita una reducción de información externa. La construcción del grafo y sus índices también se contabilizan.

## Cómo encaja Markov

El [diseño HMM existente](markov-regimes.md) propone resumir una vez por sesión un contexto de mercado mediante probabilidades de estado. Se conserva la comparación con volatilidad y dispersión continuas y con un solo estado. No se crea un detector diferente por cada activo antes de demostrar que hace falta.

El estado filtrado utiliza el prefijo disponible. El suavizado con observaciones posteriores y una ruta Viterbi reconstruida sobre todo el periodo no son entradas válidas. Las probabilidades se calculan con parámetros ajustados en el tramo autorizado. Un régimen estimado no es una etiqueta cierta de crisis, de intención humana o de causa económica.

Se separan tres componentes que pueden coexistir:

- La cadena latente describe persistencia y transiciones del contexto bajo un modelo.
- El historial de eventos contiene observaciones públicas con fechas y tipos.
- El pronóstico estima eventos o retornos futuros condicionado a lo anterior.

Un HMM con transiciones homogéneas impone permanencias geométricas. Si esa restricción perjudica la comparación, un modelo semi Markov permite duración explícita. Las probabilidades de transición condicionadas por covariables son otra variante. Ambas deben superar al HMM sencillo dentro de presupuesto. Ni la complejidad adicional ni la existencia de una fórmula probabilística acreditan menor ruido.

Para eventos en tiempo continuo, una intensidad por tipo $\lambda_c(t\mid\mathcal H_t)$ produce la pérdida de proceso puntual

$$
\mathcal L_{evento}=
-\sum_j\log\lambda_{c_j}(t_j\mid\mathcal H_{t_j^-})
+\int_{T_0}^{T_1}\sum_c\lambda_c(s\mid\mathcal H_{s^-})\,ds.
$$

La historia anterior al instante del evento evita condicionar su probabilidad a su propia observación. La integral cuenta exposición sin eventos. Inicialmente esta verosimilitud se limitaría a intervalos con historia inicial y continuidad de observación acreditadas, además de una regla explícita de inicio y censura al terminar el seguimiento. Un hueco de cobertura también altera la historia que determina intensidades posteriores. Omitir el hueco de la integral no lo corrige. Esos registros necesitarían un modelo de observación y eventos no observados, o quedarían fuera de este contraste. En horizontes discretos, cobertura insuficiente significa etiqueta desconocida, nunca negativa. Si las marcas temporales solo permiten resolver sesiones o hay muchos empates, una formulación discreta puede ser más adecuada. La intensidad no identifica causalidad entre acontecimientos.

## Dos líneas de eventos y una combinación controlada

La primera comparación utiliza el mismo codificador y tres salidas separadas: retorno residual, eventos empresariales y macroeconómicos, y eventos políticos y sociales. La representación compartida puede facilitar transferencia o producir interferencia. No se da por beneficiosa de antemano.

Se comparan cabezas separadas, representación compartida y, solo después, una integración que use probabilidades de eventos para el retorno. Las versiones independientes se comparan bajo un presupuesto total declarado. Tres redes completas no son un control de igual coste para una sola red compartida.

En una formulación auxiliar,

$$
\mathcal L=\mathcal L_{retorno}
+\lambda_e\mathcal L_{empresarial/macro}
+\lambda_p\mathcal L_{politica/social},
$$

los pesos se eligen en desarrollo dentro de la misma búsqueda presupuestada. El MAE residual sigue seleccionando al predictor principal. Las métricas de eventos se informan por separado. Una tarea auxiliar que mejora su propio resultado y perjudica el retorno no acredita mejora del objetivo principal.

Los eventos pueden coincidir en un horizonte. Varias probabilidades marginales no constituyen automáticamente una distribución conjunta de combinaciones. Se utilizarán objetivos multietiqueta cuando corresponda. Una distribución conjunta explícita requeriría modelar y evaluar sus dependencias. Compartir el codificador no basta para afirmar que esa distribución está calibrada.

Cada pérdida utiliza solo etiquetas maduras. Cerca de una frontera temporal, la falta de una etiqueta auxiliar puede enmascarar esa pérdida sin eliminar una fila válida del objetivo principal. Los recuentos por tarea y sus denominadores se conservan. La purga y los cortes consideran el horizonte y disponibilidad de cada etiqueta.

Si la cabeza de retorno recibe una predicción de evento, recibe la probabilidad emitida antes del resultado, nunca el tipo o contenido futuro verdadero. Las predicciones utilizadas para ajustar un segundo nivel se generan con cortes cronológicos internos. La versión más sencilla comparte representación y mantiene salidas separadas para evitar inicialmente esa dependencia adicional.

Añadir etiquetas de eventos cambia la supervisión, además de la arquitectura. Para atribuir el efecto de compartir representación, ambos lados del contraste deben recibir los mismos objetivos auxiliares. Comparar contra un modelo sin esas etiquetas responde a otra pregunta, el valor de la supervisión adicional, que se informará como tal.

## Cómo comprobar qué señales importan

La pregunta es el cambio en calidad y coste al retirar información, después de permitir al modelo reducido aprender con sus entradas. Una atención alta, una atribución local o una correlación no contestan por sí solas esa pregunta. Variables redundantes pueden sustituirse entre sí y variables marginalmente poco informativas pueden ser útiles juntas. El [apéndice matemático](attention-replay-mathematics.md) delimita lo que puede concluirse.

La búsqueda comienza con grupos definidos por procedencia y mecanismo, antes de seleccionar variables individuales. Incluye transformaciones derivadas, familias macro, noticias, fundamentales y representación gráfica. El criterio de selección y el presupuesto se fijan dentro de entrenamiento y validación temporal. Las pruebas externas no eligen retrospectivamente el subconjunto.

Anular una entrada de un modelo ajustado con ella mide sensibilidad a esa intervención y puede producir entradas fuera de distribución. Se complementa con un modelo reajustado para la vista reducida. Para comparar arquitectura se mantienen las mismas vistas y objetivos. Para comparar información se conserva una familia de modelos y se registran los cambios de parámetros y coste que impone la dimensión de entrada.

Un estudiante reducido entrenado directamente y otro destilado usan las mismas entradas al inferir. El profesor completo es una tercera referencia. El coste total incluye su ajuste y la generación de objetivos. La pérdida de destilación se contrasta con el MAE principal, porque copiar una media condicional no equivale a aprender su mediana.

Se registran dos fronteras: calidad con presupuesto común y coste para un margen de calidad predefinido. La no inferioridad se estima con diferencias pareadas por sesión y un margen fijado antes de evaluar. No se concluye equivalencia por obtener un valor p grande. Las familias de eventos y regímenes relevantes se definen de antemano, con tamaños e incertidumbre propios. Una media global favorable puede ocultar daño en un contexto poco frecuente.

La alternativa reducida se acepta si conserva la calidad dentro del criterio declarado y reduce un recurso medido, o mejora el error con el mismo presupuesto. Los subconjuntos con datos insuficientes se marcan como inconclusos. La información predictiva que no está en las entradas ni en la historia no aparece por repetir más veces un algoritmo.

## Ruido, confianza y situaciones nuevas

Se distinguen errores de datos, variación difícil de predecir y cambios de distribución. Fechas incorrectas, duplicados, monedas, unidades y revisiones se tratan mediante contratos de datos. Las pérdidas resistentes a valores extremos, el replay o las probabilidades de régimen no sustituyen esa auditoría.

Un evento raro y auténtico no es un error que deba suavizarse. El filtrado se ajusta con entrenamiento y se contrasta con datos sin ese filtro. No usa ventanas centradas con futuro, no modifica las etiquetas finales para mejorar la métrica y conserva la posibilidad de recuperar el dato original. Los escenarios sintéticos permiten distinguir corrupción, salto real, ruido aleatorio y cambio persistente porque su mecanismo está definido. Los datos reales no ofrecen automáticamente esa identificación.

La autoevaluación funcional puede estimar probabilidad de superar un umbral de error, cobertura de memoria o incertidumbre del pronóstico. Se comprueban discriminación, calibración y utilidad por separado. Las decisiones de abstención no eliminan los casos difíciles del MAE principal. Los intervalos o probabilidades se calibran después de seleccionar la política correspondiente.

En eventos se comparan Brier y log score frente a frecuencia histórica y calendario, además de resultados por clase y horizonte. Se informa prevalencia, cobertura del registro y número de eventos únicos. La exactitud dominada por «no ocurre nada» no acredita anticipación. Si se evalúa tiempo continuo se añaden los diagnósticos correspondientes a la intensidad y la censura.

La evaluación incluye periodos posteriores y familias de situaciones no utilizadas para ajustar el mecanismo. Sus definiciones quedan fijadas antes de consultar resultados. Una señal de incertidumbre basada en entradas reducidas también puede fallar si el cambio solo afecta a una fuente que se ha retirado. Ninguno de estos controles garantiza calidad máxima ante cualquier evento.

## Contrastes y orden de ejecución futura

| Contraste | Referencia | Modificación aislada | Criterio de descarte |
| --- | --- | --- | --- |
| E1. Eventos empresariales y macroeconómicos | Calendario, frecuencia histórica y modelo discreto sencillo | Cabeza auxiliar con las mismas entradas y cortes | No mejora probabilidades fuera de selección o perjudica el retorno |
| E2. Eventos políticos y sociales | Frecuencias y estadísticas históricas por tipo | Modelo de eventos con fuentes admitidas | Depende de preguntas retrospectivas, cobertura selectiva o ejemplos elegidos por su impacto |
| E3. Modelo conjunto | E1 y E2 separados con presupuesto total comparable | Representación compartida y objetivos comunes | Transferencia negativa o ventaja explicada por más información o capacidad |
| S1. Menos transformaciones | Vista completa | Retirar transformaciones conservando sus fuentes e historia | Pérdida fuera del margen o ausencia de ahorro total |
| S2. Menos señales externas | Modelo completo de la misma familia | Subconjunto de fuentes en todos los módulos | Daño relevante por contexto o acceso oculto a variables declaradas eliminadas |
| S3. Destilación reducida | Estudiante directo con las mismas entradas | Objetivos de un profesor temporalmente válido | Ventaja ausente o coste del profesor no justificado para la carga |

No se ejecutará el producto cartesiano de estos contrastes con todos los mecanismos de memoria. Primero se audita la etiqueta de eventos y se comprueban las referencias simples. Después se estudian representación compartida y presupuesto de señales. Su combinación con atención o replay requiere que cada componente justifique el coste.

Se mantienen las semillas, la selección temporal, el test final cerrado y la igualdad de actualizaciones donde la pregunta la exige. Se cuentan preparación de fuentes, construcción de etiquetas, codificación, filtrado, ajuste de selectores, inferencia auxiliar, recuperación, entrenamiento y evaluación. Una variante que comprime después de codificar todo no recibe un ahorro ficticio de adquisición.

La futura implementación reutilizaría el HMM C++20 existente cuando corresponda y primitivas por lotes de LibTorch, BLAS y CUDA. El modelo continuo de eventos no se añade antes de medir la referencia discreta. Las colas, índices, cachés, estados por activo y checkpoints tienen límites explícitos. Esta revisión no acredita una ganancia de latencia, VRAM o energía de las propuestas.
