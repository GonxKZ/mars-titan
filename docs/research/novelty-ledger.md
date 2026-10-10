# Antecedentes, hipótesis y criterios de aportación

Estado: propuestas contrastables. No hay una mejora experimental demostrada ni una certificación de originalidad. La búsqueda incluye fuentes primarias recientes y una revisión crítica independiente de las hipótesis.

## Hipótesis que merecen un experimento

| ID | Pregunta concreta | Antecedentes próximos | Diferencia propuesta | Qué la refutaría |
| --- | --- | --- | --- | --- |
| N1 | ¿La retención por sorpresa (error maduro, anomalía y relevancia económica), con diversidad adicional, mejora la predicción frente a un búfer uniforme del mismo tamaño? | CLS, replay, DER, Latent Replay, Titans, MIRAS y memoria financiera, incluido TRA para el uso de errores históricos. | Selección sobre predicciones realmente emitidas, anomalías calculadas con pasado y eventos publicados, con ablaciones de cada componente e igual presupuesto de bytes y escrituras. La diversidad no sustituye la anomalía. | El búfer aleatorio o uniforme iguala o mejora el error y la adaptación, o el efecto desaparece al igualar cómputo. Mejorar solo la recuperación no confirma utilidad predictiva. |
| N2 | ¿Asignar hasta cuatro pasos de lectura según información observable mejora la frontera de error y latencia? | ACT, PonderNet, Universal Transformer, recurrencia latente, RD-VLA y TTC. | Política pequeña condicionada por eventos financieros disponibles, evaluada con calibración de la política completa. | K fijo o una puerta sencilla domina en error, cobertura y p99. La convergencia latente no cuenta como confirmación. |
| N3 | ¿La selección de recuerdos y la política de recurrencia se complementan? | Combinaciones de memoria y cómputo adaptativo ya publicadas, incluido TIEM como antecedente financiero cercano. | Contraste factorial de retención básica/propuesta y puerta básica/propuesta sobre retorno residual bajo presupuesto. | La interacción no es estable entre ventanas o se explica por más actualizaciones, datos o capacidad. |
| N4 | ¿Un alumno de un paso conserva una mejora del profesor recurrente con menor latencia? | Destilación, DER y refinamiento recurrente. | Transferencia de una política numérica financiera con cortes verificables, coste total del profesor y recalibración independiente. | K = 1 directo es igual o mejor, o la ventaja de tiempo desaparece al contabilizar preparación y recuperación de memoria. |

N1 es la hipótesis principal. N2 puede entrar con una decisión temprana de continuidad. N3 solo se ejecuta si N1 y N2 justifican el coste. N4 es una extensión. Esta jerarquía protege la comparación y la entrega frente a una combinación de todos los mecanismos sin atribución posible.

## Contrastes ampliados del 4 de octubre de 2026

La [revisión de memoria, contexto y adaptación](neuroarchitecture-review.md) conserva el candidato y añade una línea separada. Estos contrastes son exploratorios hasta fijar su protocolo. No modifican las campañas en marcha ni autorizan implementar o entrenar MARS-TITAN.

| ID | Pregunta | Antecedente y diferencia que se mediría | Control y descarte |
| --- | --- | --- | --- |
| N5 | ¿Una lectura contextual aporta utilidad frente a la lectura original del mismo banco? | TRA y Financial RAG ya adaptan rutas o recuperación con errores históricos. Se mediría utilidad pareada de dos pronósticos emitidos antes de la etiqueta, con cuatro modalidades y macro. | Primero una trayectoria de memoria común anclada a la predicción base, después estados independientes para medir el sistema completo. Descartar si mezcla constante o adaptación sin contexto igualan el resultado a coste comparable. |
| N6 | ¿Separar escala del error y persistencia del cambio mejora la escritura? | Piray y Daw distinguen ruido de observación y cambio latente. La adaptación financiera es una hipótesis, no una identificación de esas causas. | Mismos candidatos, cuotas, bytes y escrituras. Contrastes de ausencia de señal, ruido, saltos y cambios persistentes. Descartar si selección uniforme o por error domina, o si el mecanismo retiene contaminación. |
| N7 | ¿Una actualización proximal de cohorte mejora calidad o estabilidad con un coste aceptable? | Operadores proximales, reglas delta y RLS son conocidos. El apéndice deriva condiciones para una escritura conjunta concreta. | Comparar con delta normalizada y una referencia pertinente de olvido. La regla cambia el algoritmo. Contracción bajo entradas fijas no demuestra mejor MAE ni estabilidad de la red completa. |

N5 se estudia antes de combinarlo con N6. N7 constituye otro eje, no una optimización silenciosa de N1. Se mantienen presupuestos y criterios predefinidos y se contabilizan generación de predicciones de desarrollo, rutas auxiliares y selección. Las [comprobaciones algebraicas](memory-mathematics.md) ejecutadas no cuentan como resultados predictivos de estas hipótesis.

La segunda ampliación del 4 de octubre concreta [atención y repetición](attention-replay-review.md) y [eventos y presupuesto de señales](event-signal-comparison.md). Conserva N1 como pregunta principal. Los contrastes siguientes no cambian la campaña activa y necesitan un registro independiente antes de ejecutarse.

| ID | Pregunta | Diferencia que se mediría | Control y descarte |
| --- | --- | --- | --- |
| N8 | ¿Separar las repeticiones mejora retención y error futuro con las mismas exposiciones? | Ordenar el mismo multiconjunto de episodios maduros, antes de introducir un calendario adaptativo o prioridades. SRT y la literatura de recuperación ya estudian cuándo repetir. | Igualar ejemplos, exposiciones, actualizaciones y bytes. Descartar si el beneficio se limita al búfer o empeora MAE futuro. La selección por utilidad y la corrección por importancia forman otro eje. |
| N9 | ¿Aprender eventos públicos aporta señal útil al retorno residual? | Comparar eventos empresariales y macroeconómicos, políticos y sociales, separados y con representación compartida. Daily Oracle, MIRAI, StockMem y Hawkes Attention aportan antecedentes con tareas distintas. | Mismas fuentes, objetivos auxiliares y presupuesto al comparar arquitectura. Separar ocurrencia, contenido y retorno. Descartar si la ganancia procede de información retrospectiva o perjudica al objetivo principal. |
| N10 | ¿Puede una vista reducida conservar calidad con menor coste? | Distinguir compresión, transformaciones redundantes, retirada de fuentes y destilación. Auditar todos los módulos, incluido HMM, memoria y calibración. | Misma cohorte admitida completa, modelo reducido reajustado y margen de no inferioridad predefinido. Descartar si hay acceso indirecto a señales eliminadas, daño relevante por contexto o ahorro solo aparente. |

N10 define un contraste de información distinto de las ablaciones internas que mantienen todas las entradas. La referencia completa conserva las cuatro modalidades y macro. Ninguna variante reducida amplía su población relajando admisión. El [apéndice de repetición y señales](attention-replay-mathematics.md) explica por qué más exposiciones no crean hechos nuevos y por qué retirar información no puede garantizar calidad ante cualquier evento.

## Diseño mínimo para N1 y N2

Se conservarán el mismo codificador, entradas, objetivo residual, particiones y presupuesto de búsqueda. Para N1 se comparan ausencia de memoria, memoria uniforme, selección solo por error y selección por sorpresa completa. Esta última conserva error maduro, anomalía y relevancia económica conforme a O3. La diversidad se estudia por separado como criterio adicional, con escalas y pesos fijados en desarrollo. Para N2 se comparan K = 1, 2 y 4, además de puertas solo si los pasos adicionales muestran valor durante desarrollo.

La especificación selectiva utiliza tres índices y un almacén único de episodios. Su puntuación de admisión se conserva, de modo que no se interpreta como una optimización continua de diversidad global. La capacidad se iguala en bytes, contabilizando claves, valores y referencias. Las claves fijas y los valores proyectados al leer evitan mezclar representaciones entrenables antiguas en el núcleo. Estas decisiones también deben declararse al compararlo con una memoria de claves aprendidas.

Los contrastes principales se fijan antes del test. El resto se identifica como exploratorio. El error se compara de forma pareada por sesión y la incertidumbre mantiene los activos de la misma fecha juntos. La variable macro o el régimen no se seleccionan retrospectivamente para hacer aparecer una mejora.

K cuenta lecturas y refinamientos, sin una lectura extra en la inicialización. La comparación entre banco episódico y memoria neural de pesos rápidos mantiene identificada cada variante. La revisión de antecedentes exige esa referencia neural antes de atribuir una aportación frente a Titans. Se registran candidatos de escritura, operaciones de índices, valores únicos y cómputo efectivo de cada celda.

En un diseño factorial, la interacción puede escribirse como diferencia entre dos efectos:

$$\Delta_{int}=(L_{11}-L_{10})-(L_{01}-L_{00}),$$

donde L es pérdida fuera de muestra, el primer índice indica retención y el segundo política de pasos. Su signo y tamaño se interpretan junto a la incertidumbre y los recursos. No representa un efecto causal económico. Si los intervalos admiten mejoras y empeoramientos relevantes, la conclusión es incierta.

## Deducciones de diseño

**Disponibilidad por composición.** Si parámetros, transformaciones, índices y estado inicial se han construido con información permitida, cada lectura solo accede a esa información y cada actualización utiliza eventos ya conocidos, la salida sigue dependiendo del pasado permitido. Se comprueba por inducción sobre actualizaciones y pasos internos. Es una propiedad condicional de la especificación, no una prueba de que los datos reales cumplan las premisas.

**Información frente a cómputo.** Repetir una transformación de las mismas entradas y memoria no añade una observación externa. Puede mejorar la aproximación o la recuperación que realiza la red. Por tanto, el bucle tiene sentido como presupuesto de cálculo, pero no como garantía de eliminar incertidumbre del mercado. Esta distinción es conocida y no se reivindica como teorema nuevo.

**Memoria activa frente a corpus.** Con capacidad de episodios, colas y lote fijadas, los tensores de trabajo no necesitan crecer con todos los registros históricos. El estado por activo sí crece con el número de activos, salvo que se pagine. La cota debe incluir copias y optimizadores. De esta forma se puede diseñar un recorrido mayor que la RAM sin afirmar tiempo constante ni memoria infinita.

Estas deducciones ayudan a construir pruebas de invariancia y de consumo. El conocimiento nuevo, si lo hay, dependerá del resultado de aplicar y contrastar el diseño en el problema financiero.

## Cómo se controla la afirmación de novedad

Para cada comparación final se conservarán el artículo más cercano, su versión, objetivo, datos, política de memoria, regla de adaptación, calibración y coste. Se comprobarán especialmente SFM, DoubleAdapt, TRA, FinMem, FinAgent, FinCon, MacroHFT y TIEM, además de los trabajos de memoria y recurrencia general.

[TRA](https://arxiv.org/html/2106.12950v2) impide presentar el uso de errores históricos para enrutar predicciones como una novedad general. Para O3 y O4 se distinguirá el efecto del régimen del efecto de recuperar errores, si esa entrada se incorpora. La incorporación bibliográfica no exige reproducir TRA ni ampliar el núcleo. El diseño seguirá contrastando retornos residuales con disponibilidad temporal y recursos comunes.

Un componente basado en código ajeno se identifica y conserva su licencia. Una adaptación se describe mediante sus cambios verificables. La frase «no se ha probado antes» solo se sustituiría por una afirmación limitada al alcance y fecha de una búsqueda documentada, nunca por una garantía universal.

El registro de ensayos incluye intentos manuales, modelos descartados, búsquedas de variables y decisiones de alcance. Obtener un resultado favorable tras muchos intentos requiere un análisis que tenga en cuenta ese proceso. El test final no se reutiliza para inventar una explicación o diseñar otra variante.

## Resultado científicamente útil

Una mejora pequeña que resista controles, una reducción de coste sin pérdida relevante de calidad o un límite bien caracterizado pueden aportar evidencia útil. Si una referencia sencilla domina, se conserva ese resultado y se explica. La selección final no está condicionada a que el modelo más complejo gane.

## Búsqueda de precedentes del 9 y 10 de octubre de 2026

Las [propuestas posteriores a Titans](post-titans-proposals.md) pasaron por una búsqueda declarada antes de redactar su estado de novedad. El 9 de octubre se consultaron la API de arXiv, la API v2 de OpenReview, la API de Semantic Scholar y la búsqueda de repositorios de GitHub con 25 consultas agrupadas por propuesta, con sinónimos de comunidades vecinas (control adaptativo, filtrado, sistemas conmutados, neurociencia computacional y datos de panel). Semantic Scholar respondió con HTTP 429 en la mayoría de consultas y esas consultas cuentan como no ejecutadas en esa fuente. El 10 de octubre se añadieron 12 consultas de seguimiento en arXiv, la revisión completa de los títulos de PMLR v306 (ICML 2026) y la lectura de los resúmenes de los trabajos más cercanos. El [registro de la búsqueda](../../reports/research/post-titans-precedent-search-20261010.json) conserva consultas, totales y resultados revisados. Una búsqueda vacía o sin coincidencias no demuestra que no exista un precedente.

| ID | Propuesta | Trabajos más cercanos y diferencia exacta | Estado |
| --- | --- | --- | --- |
| N11 | PT1, puertas de Titans en una caja certificada y escritura recortada | Cao et al. (2026, ICML) dan condiciones de Lyapunov cuadrática común para SSM selectivos y extraen certificados de los pesos. MDN (Huang et al., 2026, ICML) restringe el espectro de cada paso de una regla delta con momentum. Jin et al. (2026, preprint) analizan LMS con momentum constante bajo excitación estocástica. MIRAS propone Huber como sesgo atencional. Ninguno reduce la memoria de Titans a dos bloques 2 × 2 para claves arbitrarias, da la condición necesaria y suficiente ni parametriza las puertas dentro de una caja certificada con contracción incremental bajo recorte. | No encontramos precedente en la búsqueda del 9 de octubre de 2026 con estas consultas para esa combinación concreta. Las técnicas (LMI en vértices, modificación σ y Huber) son conocidas y se citan. |
| N12 | PT2, conformal en línea con etiquetas maduras por cohortes | ACI y DtACI (Gibbs y Candès), seguimiento de cuantiles (Angelopoulos et al., 2023), ACI multipaso (Szabadváry, 2024, y Wang y Hyndman, 2024) y ACI con retroalimentación retrasada (Halabi y Brandt, 2026). | Adaptación con precedentes directos. No se reivindica novedad. |
| N13 | PT3, regla de Kalman para B6 con ruido de cohorte equicorrelacionado | Filtro de Kalman y modelos lineales dinámicos, efectos temporales aleatorios en paneles y el banco de RLS de Fentazi et al. (2026) con etiquetas retrasadas. | Adaptación. Los componentes son clásicos. No encontramos precedente de su combinación para cohortes maduras de rentabilidades residuales en la búsqueda del 9 y 10 de octubre con estas consultas, pero no se presenta como aportación metodológica. |
| N14 | PT4, memoria de mercado compartida con escritura por gradiente medio de la sesión | Titans-QFWP usa una memoria de estilo Titans para cartera, con producto exterior y sin pérdida asociativa. TRA y los modelos transversales usan información entre activos sin pesos rápidos compartidos. | No encontramos precedente en la búsqueda del 9 de octubre de 2026 con estas consultas. Línea posterior sin implementar. |
| N15 | PT5, cascada de Benna y Fusi sobre pesos rápidos | Kaplanis et al. (2018) la aplican a parámetros de una DQN. Nested Learning actualiza bloques a varias frecuencias sin acoplamiento bidireccional. Memini (2026) la usa en aristas de un grafo para LLM sin experimentos. | Adaptación. No encontramos su uso sobre la memoria de Titans en la búsqueda del 9 y 10 de octubre, y el control obligatorio es un banco de exponenciales con el mismo estado. |
| N16 | PT6, modulación de la fusión por patrón de presencia | FiLM y numerosos trabajos de robustez a modalidades ausentes. | Adaptación con precedentes directos. |

Estas formulaciones se limitan al alcance y la fecha de la búsqueda. Si aparece un precedente, la propuesta se presenta como adaptación y se cita, sin cambiar sus controles ni su regla de decisión.
