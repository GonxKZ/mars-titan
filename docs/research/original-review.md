# Revisión crítica de la propuesta técnica original

Autor del proyecto: Gonzalo García Lama. Fecha de revisión: 18 de septiembre de 2026.

Esta revisión describe el antecedente técnico. El estado de implementación y los
experimentos posteriores se recogen en los [informes](../../reports/README.md).

## Documentos y criterio de revisión

Se revisa íntegramente el [documento inicial conservado localmente](source/README.md), fechado el 8 de mayo de 2026, como antecedente técnico propio del proyecto. Tiene 45 páginas PDF: ocho preliminares y 37 páginas de cuerpo, anexos y bibliografía. En este documento se cita primero la página PDF y, cuando corresponde, la página impresa entre paréntesis.

La revisión toma como referencia los seis objetivos del proyecto, recogidos en la [hoja de ruta](roadmap.md), y el [protocolo de investigación](protocol.md). El alcance es una comparación experimental de alternativas para predecir retornos residuales con las mismas entradas y condiciones de evaluación.

La revisión evalúa la coherencia interna del original y su adecuación al alcance presentado. Las afirmaciones bibliográficas o cifras del dataset que aparecen en él no se consideran verificadas por el mero hecho de estar escritas: su contraste con las fuentes primarias y con los archivos reales forma parte del trabajo posterior.

## Valoración general

El original proporciona una base conceptual útil: formula una pregunta contrastable, reconoce que las gráficas representan expectativas, sitúa la disponibilidad temporal de los datos antes que la complejidad del modelo y propone referencias simples, ablaciones, incertidumbre y análisis de fallos. Su portada y resumen declaran expresamente que se trata de una propuesta metodológica y no de resultados empíricos. El anexo A conserva todos sus controles como pendientes (PDF pp. 1-2 y 42. P. impresa 34).

Para ejecutar la comparación necesita acotar el alcance y definir sus componentes con precisión. Acumula memoria por regímenes, sorpresa económica, fusión multimodal, adaptación local, protección frente al olvido, abstención, decisiones de cartera y controles de producción sin demostrar aún que puedan implementarse y compararse con el presupuesto disponible. La contribución será el conocimiento obtenido mediante una comparación reproducible, incluidos los casos en que la arquitectura propuesta no mejore a referencias más sencillas.

No procede tratar el documento original como una memoria final. Tampoco es necesario desecharlo: sus hipótesis, riesgos y mecanismos sirven como antecedentes que deben convertirse en definiciones precisas, experimentos ejecutables y resultados verificables.

## Correspondencia con la propuesta presentada

| Objetivo de la propuesta | Aportación del original | Ajuste necesario para la comparación |
| --- | --- | --- |
| O1. Auditar precios, noticias, fundamentales y representaciones visuales de FinMultiTime, respetando su disponibilidad temporal | Describe frecuencias heterogéneas, riesgos de fuga y una tupla con tiempo de evento y de disponibilidad. Propone contratos de datos. PDF pp. 11-12, 24 y 35 (impresas 3-4, 16 y 27). | Convertir las reglas generales en un inventario auditado. Documentar cobertura real, procedencia, licencia, zona horaria, calendario, fecha de publicación y exclusiones por modalidad. Las cifras globales del original son afirmaciones de su fuente, no el tamaño ya comprobado del subconjunto experimental. |
| O2. Formular retornos residuales de mercado y, si es posible, sector | La ecuación 4.10 propone descontar mercado y sector. PDF p. 17 (impresa 9). | Definir horizonte, estimación pasada de coeficientes y momento de disponibilidad de cada etiqueta. Mantener el ajuste sectorial condicionado a la disponibilidad y calidad de los datos, como en la propuesta presentada. |
| O3. Diseñar memoria de eventos con sorpresa por error, anomalía y relevancia económica, régimen e incertidumbre | Presenta sorpresa compuesta, bancos de memoria, protección de retención y salidas probabilísticas. PDF pp. 18-23 (impresas 10-15). | Comenzar por una formulación implementable de estos elementos. Resolver la actualización con etiquetas diferidas y distinguir mecanismos centrales de extensiones como autoediciones, proyecciones de retención o numerosas cabezas de decisión. |
| O4. Comparar modelos clásicos y neuronales y ablaciones sin memoria, global y sin sorpresa | Incluye una lista extensa de referencias y nueve ablaciones. PDF pp. 27-28 (impresas 19-20). | Definir un conjunto manejable y justificarlo. Igualar datos, particiones y oportunidades de ajuste. Separar reproducción publicada, adaptación propia y referencia conceptual. Conservar como mínimo las ablaciones de la propuesta presentada. |
| O5. Evaluar MAE/MSE, dirección, calibración, Rank IC y simulación long-short con costes, drawdown y turnover en walk-forward | Enumera métricas y propone bloques temporales, embargo, costes y varias semillas. PDF pp. 24-26 (impresas 16-18). | Especificar ventanas recurrentes, reglas de selección, estado de la memoria, etiquetas solapadas, calibración y simulación. Elegir de antemano la medida principal y cómo expresar la incertidumbre de las diferencias. |
| O6. Analizar fallos, modalidades y coste computacional con 8 GB de VRAM | Incluye un premortem, limitaciones y un presupuesto de recursos propuesto. PDF pp. 13, 30-36 y 40 (impresas 5, 22-28 y 32). | Medir recursos reales y explicar qué aportan las modalidades. Usar el premortem para diseñar controles. No presentarlo como diagnóstico de fallos observados ni como acreditación de preparación para producción. |

El cuadro de fases del original no coincide con esta secuencia: sitúa los baselines en la fase 2 y la auto-adaptación en la fase 5 (PDF p. 37. Impresa 29). La planificación debe seguir O1 datos, O2 residuales, O3 memoria e incertidumbre, O4 referencias y ablaciones, O5 evaluación y O6 análisis crítico. Estos identificadores organizan el trabajo y su trazabilidad, no la estructura de carpetas del repositorio.

## Correcciones metodológicas prioritarias

### 1. Evitar que la memoria acceda a etiquetas futuras

La definición del recuerdo incluye como valor un «efecto residual futuro» (ecuación 4.13, PDF p. 19. Impresa 11). La sorpresa incorpora error económico y regret (PDF p. 18. Impresa 10), y el pseudoalgoritmo predice, calcula pérdidas y actualiza memoria dentro del mismo lote sin indicar cuándo se han observado esos resultados (PDF pp. 22-23. Impresas 14-15). El problema no es usar retornos futuros como etiquetas supervisadas. Es que esa redacción deja abierta su utilización antes de que estén disponibles en la simulación temporal.

Cada observación debe distinguir el instante de decisión, el final del horizonte y el instante de disponibilidad de la etiqueta. Una predicción en `t` solo puede leer información y actualizaciones disponibles hasta `t`. Las actualizaciones que dependan de error, regret o retorno realizado deben esperar a que madure la etiqueta. El protocolo debe especificar el orden entre predicción y actualización, procesar los eventos cronológicamente y evitar que el orden de activos de un mismo instante introduzca información de resultados todavía desconocidos.

También deben fijarse el estado inicial de la memoria y su reinicio o reconstrucción en cada partición y recorrido de entrenamiento. Reutilizar una memoria que ya ha pasado por fechas posteriores puede invalidar incluso un conjunto de características correctamente construido. El conjunto usado para medir retención debe contener únicamente información permitida por el protocolo en ese instante.

### 2. Concretar la disponibilidad de modalidades y el objetivo residual

El original formula correctamente el principio `t_available ≤ t_decision` (PDF p. 24. Impresa 16), pero la representación de noticias se filtra solo por `τ ≤ t`, mientras que los fundamentales utilizan explícitamente disponibilidad (ecuaciones 4.3-4.4, PDF p. 17. Impresa 9). Esa diferencia debe resolverse con una definición común y verificable. La fecha de una noticia, un periodo contable o el nombre de un archivo visual no bastan por sí solos para acreditar disponibilidad.

Para cada modalidad se necesita una regla específica: publicación efectiva y zona horaria para noticias. Fecha de publicación y tratamiento de revisiones para fundamentales. Extremo final y momento de disponibilidad para ventanas de gráficos. Ajustes corporativos y calendario para precios. Si una fecha no puede acreditarse, debe excluirse ese uso o marcarse un supuesto conservador con sus límites. No puede afirmarse que se ha eliminado toda fuga solo porque un campo se llame `available_at`.

El retorno futuro de mercado o sector puede formar parte de la etiqueta residual, pero no de las características conocidas en la decisión. Los coeficientes de exposición deben estimarse con información pasada. La definición debe aclarar ventanas, historia mínima y relación entre factores de mercado y sector. Las transformaciones y normalizaciones deben ajustarse en la información permitida por cada ventana.

### 3. Delimitar el significado de «causal» y «contrafactual»

El original utiliza «sorpresa causal», «contexto causal», «memoria contrafactual» y «ablaciones para demostrar causalidad arquitectónica» (PDF pp. 16-20 y 28. Impresas 8-12 y 20). El diseño presentado no define una estrategia de identificación causal de efectos económicos ni los supuestos necesarios para interpretarlos como contrafactuales.

Respetar el orden temporal impide utilizar futuro, pero no elimina por sí solo las variables de confusión. Descontar retornos de mercado y sector produce un residual de un modelo, sin identificar automáticamente el efecto de una noticia o evento. Las ablaciones permiten estimar la contribución de un componente bajo el experimento realizado. No demuestran que el evento cause el rendimiento financiero.

La memoria final puede describirse como memoria temporalmente válida de eventos, memoria de retornos residuales o adaptación por régimen. Los nombres originales pueden conservarse como antecedentes o etiquetas del diseño, acompañados de una definición precisa y de esta limitación. No se atribuirán efectos causales o contrafactuales identificados a los resultados de la comparación.

### 4. Definir una arquitectura mínima reproducible

Las ecuaciones de lectura y actualización son esquemáticas. No se define de forma suficiente cómo construir el subespacio de protección `P⊥R`, estimar `RetentionRisk`, seleccionar recuerdos protegidos, gestionar capacidad o elegir umbrales (PDF pp. 18-20. Impresas 10-12). El conjunto de bancos mezcla regímenes de mercado, volatilidad, eventos de resultados y sectores (ecuación 4.14, PDF p. 19. Impresa 11). Esas categorías pueden solaparse y no constituyen automáticamente una partición homogénea.

Antes de implementar hay que fijar representación, claves y valores, capacidad, lectura, escritura, olvido, enrutamiento, normalización de las componentes de sorpresa y aprendizaje de sus pesos. Debe aclararse qué partes se entrenan fuera de línea, cuáles cambian durante la evaluación y con qué datos. Una adaptación propia inspirada en TITANS o SEAL debe nombrarse como tal. La referencia bibliográfica no acredita una reproducción del método publicado.

El núcleo de O3 puede evaluarse con una memoria sencilla y trazable, una medida explícita de sorpresa y una salida probabilística. La protección de retención más compleja y las autoediciones del original (PDF pp. 20 y 22-23. Impresas 12 y 14-15) deben añadirse solo si pueden formalizarse, implementarse y compararse sin comprometer O4-O6. Son extensiones candidatas, no resultados ni obligaciones académicas.

### 5. Sustituir el reparto de años por un protocolo completo de evaluación

El original denomina walk-forward a una propuesta de entrenamiento 2009-2018, validación 2019-2020 y dos pruebas 2021-2022 y 2023-2025 (PDF p. 24. Impresa 16). Esos bloques describen una separación temporal, pero no especifican cómo avanzan las ventanas, cuándo se vuelve a entrenar, qué se congela ni qué información se permite reutilizar. La cobertura de esos años debe comprobarse por activo y modalidad.

El protocolo debe definir las ventanas de entrenamiento, validación, calibración y prueba, junto con la política de reentrenamiento y el tratamiento del estado de memoria. También debe especificar cómo se purgan las observaciones cuyas etiquetas invaden el periodo siguiente y qué separación temporal requiere el horizonte. La selección de modelos, umbrales y costes no puede depender del periodo final de prueba. Si se permite adaptación durante la prueba, debe estar prevista y usar únicamente datos ya observados. Su evaluación se distinguirá de la evaluación con modelo congelado.

La comparación necesita semillas y ventanas identificadas, presupuesto de ajuste documentado e intervalos que respeten la dependencia temporal y entre activos. Tratar todas las filas de activo-fecha como observaciones independientes puede ofrecer una precisión estadística engañosa. Las pruebas entre mercados o sectores que el original propone son extensiones posibles, condicionadas a cobertura y presupuesto.

### 6. Hacer justa la comparación y medir el presupuesto de recursos

La lista del original abarca modelos lineales, árboles, CNN, GRU, LSTM, modelos multimodales y varias memorias neurales (PDF p. 27. Impresa 19). Es una reserva de alternativas, no un compromiso realista de reproducirlas todas. La comparación principal debe seleccionar representantes suficientes para responder a la pregunta y las ablaciones requeridas por la propuesta: sin memoria, memoria global y sin sorpresa. Los cambios de capacidad, modalidad o presupuesto deberán registrarse para no atribuirlos indebidamente a la memoria.

El objetivo de 185-255 millones de parámetros, las dimensiones de memoria y la distribución propuesta de VRAM no se apoyan todavía en una medición (PDF pp. 13 y 37-38. Impresas 5 y 29-30). La viabilidad depende también de activaciones, optimizador, precisión, longitud, lotes y coste de obtener representaciones. Debe partirse de un modelo compacto medido, con registro de GPU, pico de VRAM y tiempo. El límite de 8 GB es una restricción experimental. No se confunde con una afirmación sobre la capacidad física del equipo.

Si un modelo no cabe o no puede reproducirse bajo el presupuesto, debe constar como exclusión justificada. Una implementación reducida o adaptada debe identificarse sin presentar sus resultados como una reproducción exacta. El cómputo de preprocesamiento y extracción de representaciones también forma parte del coste de la solución, aunque se ejecute por separado.

### 7. Precisar métricas, incertidumbre y simulación económica

El original enumera numerosas métricas y combina objetivos predictivos, económicos y de retención (PDF pp. 21-23 y 25-26. Impresas 13-15 y 17-18). Esa amplitud necesita una jerarquía para evitar seleccionar después del experimento la métrica que favorezca al modelo. La medida principal, las secundarias y las reglas de comparación deben quedar fijadas antes de analizar la prueba final.

Para retornos continuos, la calibración debe vincularse a la distribución o a los intervalos predichos y a su cobertura. Una métrica como ECE necesita probabilidades de eventos o clases bien definidos. Los umbrales de abstención y sus relaciones entre cobertura, error y utilidad se ajustarán fuera del test. Las diferencias de rendimiento deben acompañarse de magnitud, variación entre ventanas y límites de incertidumbre, no solo de promedios.

La simulación long-short necesita una regla de cartera explícita, instante de decisión y de ejecución, horizonte, rebalanceo, tratamiento de posiciones solapadas, costes, rotación y restricciones relevantes. Si las características usan el cierre, no debe darse por supuesto que se ejecuta a ese mismo precio sin justificar la cronología. Los resultados son los de una simulación retrospectiva bajo supuestos documentados. Un Sharpe neto positivo no acredita ejecución real ni rentabilidad futura.

### 8. Separar calidad de la investigación de superioridad del modelo

El original presenta como criterio de éxito una mejora de Rank IC y un Sharpe neto positivo (PDF p. 29. Impresa 21). A la vez, reconoce que el proyecto puede aportar conocimiento si la memoria no mejora a referencias sencillas y si se documentan correctamente los fallos (PDF pp. 39-40. Impresas 31-32). Esta segunda posición es la adecuada para valorar la investigación.

Los criterios de validez del estudio deben exigir trazabilidad, comparación justa y ausencia de fugas conocidas. Los criterios de apoyo a la hipótesis deben exigir las diferencias que se hayan definido. Son cuestiones distintas: una comparación válida puede refutar la hipótesis. Un resultado favorable obtenido con fuga no la respalda. También debe distinguirse un resultado no concluyente de una demostración de equivalencia.

La recomendación del original de eliminar cualquier módulo que no mejore debe aplicarse durante el desarrollo o validación, no mediante iteraciones guiadas por el test. En la memoria se conservarán los resultados de ablaciones y los intentos descartados que sean relevantes para comprender la conclusión.

## Bibliografía y presentación

La bibliografía original contiene ocho entradas y utiliza citas numéricas (PDF p. 45. Impresa 37). El informe debe mantener un estilo de citación uniforme. Las entradas [5]-[8] aparecen en la bibliografía del texto extraído sin llamadas localizadas en el cuerpo, de modo que deben revisarse su uso y pertinencia.

La cobertura bibliográfica debe ampliarse y verificarse en los aspectos que sostienen el método: evaluación temporal, incertidumbre, aprendizaje con memoria, comparación de predictores y simulación económica. No se fija un número mínimo arbitrario de referencias. Cada fuente debe contribuir a una decisión o argumento, y las diferencias frente al método publicado deben describirse con precisión.

La portada original no identifica nominalmente al autor y presenta el trabajo como documento científico en LaTeX (PDF p. 1). El informe final debe identificar a Gonzalo García Lama y distinguir la versión del documento y el periodo cubierto por los experimentos.

Los capítulos de premortem y sistema «production-ready» contienen controles útiles de integridad y trazabilidad (PDF pp. 30-36. Impresas 22-28), pero desplazan el foco si se mantienen como promesa de despliegue. Conviene integrarlos en amenazas a la validez, pruebas de robustez y trabajo futuro. El alcance actual no incluye operar en un mercado real.

Las figuras de presupuestos y mejoras esperadas deben seguir claramente etiquetadas como esquemas o hipótesis. La memoria final sustituirá las figuras de expectativas que no sean necesarias por gráficos generados desde experimentos reales. No deben emplearse valores ilustrativos en tablas de resultados ni narrarse riesgos del premortem como si ya hubieran ocurrido.

El original se conserva como antecedente. Sus afirmaciones y figuras no se convierten en resultados observados al incorporarlas al informe. Los métodos y materiales de terceros mantienen su atribución y sus condiciones de uso.

## Decisión de revisión

Se conserva la pregunta comparativa, la prioridad del dato disponible en cada instante, el objetivo residual, la memoria de eventos con sorpresa e incertidumbre, las referencias simples, las ablaciones y el análisis de fallos. Se concreta el alcance en los seis objetivos de la propuesta y se reserva la complejidad adicional para extensiones justificadas.

Antes de poder afirmar que el trabajo cumple su contribución empírica deben estar disponibles un subconjunto auditado, un protocolo temporal cerrado, implementaciones comparables, resultados reproducibles y una discusión proporcionada. La redacción de conclusiones, la validación de 8 GB y cualquier afirmación de mejora permanecen pendientes de esos experimentos.
