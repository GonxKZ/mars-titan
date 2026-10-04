# Memoria, contexto y adaptación para una variante ampliada

Fecha de corte: 4 de octubre de 2026. Estado: investigación y diseño. MARS-TITAN sigue sin implementar ni entrenar. El [candidato existente](candidate-architecture.md) se conserva como referencia. Esta revisión propone cambios separados que podrían descartarse sin alterar ese diseño ni las campañas comparativas en ejecución.

La primera ampliación que merece contrastarse es una lectura contextual con utilidad medida a partir de predicciones realmente emitidas. Después se estudiaría distinguir error, ruido y cambio persistente al escribir memoria. Una actualización asociativa proximal constituye otro contraste, con condiciones matemáticas explícitas. Compresión, más especialistas y cómputo adaptativo quedan condicionados a que esos mecanismos aporten una mejora medible.

Esta secuencia no sustituye N1, la hipótesis principal sobre selección de memoria del [registro de hipótesis](novelty-ledger.md). Delimita una línea ampliada, posterior y atribuible. Las referencias recientes proporcionan mecanismos y objeciones. Ninguna demuestra que esta combinación sea superior en mercados financieros.

## Alcance y trazabilidad de la revisión

Se han contrastado fuentes primarias sobre memoria neuronal, recuperación, aprendizaje con feedback tardío, contexto financiero, neurociencia y planificación humana. Las versiones, metadatos y pasajes consultados están en el [catálogo nuevo](../references/neuroarchitecture-sources.json), junto con su [BibTeX](../references/neuroarchitecture.bib). Se incorporan 27 referencias, además de revisar antecedentes ya catalogados. Es una revisión dirigida, no un censo exhaustivo ni una lectura integral de cada artículo.

ATLAS ya tiene [actas ICML 2026](https://proceedings.mlr.press/v306/behrouz26b.html). Se actualiza su ficha y se distingue el PDF previamente conservado de la edición editorial. Gated DeltaNet-2, el trabajo sobre enrutamiento contextual y Financial RAG se citan por sus versiones de preprint. En *When Agents Trade* se conserva el preprint leído. Los autores declaran una publicación posterior con otro título y orden de autoría, pero su DOI editorial no pudo corroborarse. La corrección publicada del estudio de navegación experta queda vinculada a su ficha.

El objetivo principal continúa siendo predecir retornos residuales con cuatro modalidades y macro. La simulación financiera es una evaluación adicional. El diseño mantiene precios, texto, fundamentales y gráficos, los 140 indicadores admitidos, disponibilidad temporal, revisiones, máscaras y versiones de codificadores. No utiliza sintéticos para reparar una ausencia de evidencia real.

## Qué aporta la frontera de arquitecturas

No hay un único modelo de referencia que domine tareas, escalas y equipos distintos. La comparación pertinente separa cómo se conserva información, qué se actualiza y cuánto cuesta recuperar una observación útil.

| Antecedente | Mecanismo pertinente | Consecuencia para la comparación |
| --- | --- | --- |
| [Titans, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) | Memoria neuronal actualizada mediante una pérdida asociativa interna, con momento y olvido. | Su sorpresa interna no es el error futuro del predictor. Conservar una referencia de pesos rápidos además del banco episódico. Sus experimentos temporales no demuestran ventaja financiera. |
| [TTT, ICML 2025](https://proceedings.mlr.press/v267/sun25h.html) | Estado definido por una actualización de aprendizaje y formulación por bloques. | La forma dual puede acelerar el mismo algoritmo de bloque. Cambiar el tamaño del bloque cambia dónde se evalúan gradientes, por lo que no siempre es una optimización semánticamente neutra. |
| [MIRAS, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d55f39791f04745b2e0c8abebf3dd5d7-Abstract-Conference.html) | Separa memoria, objetivo asociativo, retención y optimización. Incluye alternativas a la pérdida cuadrática. | Cambiar una pérdida por Huber o una retención por KL no constituye por sí solo una novedad. Variar un eje cada vez. |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) y [Gated DeltaNet-2, v1](https://arxiv.org/html/2605.22791v1) | Borrado, escritura y regla delta. La segunda separa puertas por canal. | Mayor expresividad no implica contracción de cada actualización. El apéndice comprueba un contraejemplo. No se deduce de él que el modelo publicado diverja. |
| [ATLAS, ICML 2026](https://proceedings.mlr.press/v306/behrouz26b.html) | Optimiza memoria utilizando una ventana de contexto anterior, además del elemento actual. | Comparar contra replay corto y contar caché, actualizaciones y operaciones adicionales. Una capacidad asociativa teórica bajo supuestos sobre claves y pérdidas no mide calidad predictiva ni «información por neurona». |
| [Nested Learning y HOPE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) | Optimización a varias escalas temporales. | Cada escala añade estado, frecuencia, optimizador y recuperación. La consolidación paramétrica no se introduce durante evaluación como una simple operación de caché. |
| [Memory Caching, ICML 2026](https://proceedings.mlr.press/v306/behrouz26a.html) | Conserva memorias de segmentos y combina o selecciona su lectura. | Contar todas las memorias residentes, sus índices y las escrituras. Una caché que crece con la secuencia no satisface por sí sola un presupuesto fijo. |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | Revisa discretización y expresividad de modelos de espacio de estados. | Es un candidato de referencia recurrente, no una mejora local demostrada. Las mediciones con H100 y grandes lotes no predicen el tiempo de un modelo pequeño en la RTX 4070. |
| [Engram](https://arxiv.org/abs/2601.07372v2) y [Memory³](https://arxiv.org/abs/2407.01178v1) | Separan parte del conocimiento almacenado y el cálculo del modelo. | Una tabla aprendida de n-gramas no es un banco de episodios actualizado con etiquetas maduras. Almacenes externos de gran tamaño también consumen memoria, E/S y tiempo de construcción. |
| [PEER, 2024](https://arxiv.org/abs/2407.04153v1) | Selección dispersa de muchos expertos mediante claves producto. | Pocos expertos activos no equivalen a pocos parámetros almacenados. El coste incluye router, todos los pesos, optimizadores y carga, no solo FLOPs activos. |

La clave estable de Engram conserva los autores de su preprint v2. La edición ACL consultada usa otra versión y denominación. No se mezclan metadatos de ambas para fabricar una cita única. Los detalles de lectura de las arquitecturas ya existentes se complementan con las revisiones de [sistemas](../references/systems-review.md) y [antecedentes recientes](../references/frontier-review.md).

Para el equipo local, el orden razonable es GRU y modelo sin memoria, banco uniforme, selector propuesto y estado asociativo sencillo. TTT, una regla delta con puertas o un modelo de espacio de estados entrarían como comparadores compactos si permiten una reproducción suficiente. Una implementación parcial se identificaría como adaptación, sin atribuirle el nombre de una reproducción completa del artículo.

## Qué trasladar de neurociencia y experiencia

El candidato ya distingue parámetros lentos, memoria persistente y estado de trabajo. La analogía con sistemas complementarios de aprendizaje tiene antecedentes recogidos en la [revisión del cerebro](../references/brain-review.md). Repetir esa separación no añade novedad. Las fuentes nuevas ayudan a precisar decisiones que sí pueden contrastarse.

**Aprender de un error exige distinguir su origen.** [Piray y Daw, 2024](https://www.nature.com/articles/s41467-024-53459-z) comparan aprendizaje humano cuando varían el proceso latente y el ruido de observación. Los efectos sobre la tasa de aprendizaje son distintos. Esto motiva comparar error maduro, error escalado y evidencia acumulada de cambio. No permite llamar «volatilidad del proceso» a cualquier volatilidad bursátil observada ni identificar perfectamente ambas fuentes en datos reales. El apéndice muestra dos modelos con la misma varianza de innovación y distinta ganancia óptima.

**La memoria comprimida necesita conservar sus errores y referencias.** [Spens y Burgess, 2026](https://www.nature.com/articles/s41467-026-74357-6) estudian un modelo computacional de memoria episódica y semántica mediante recuperación y compresión. También analizan distorsiones. Para finanzas, una clave compacta puede localizar un episodio, pero sus cifras, fechas y revisiones deben recuperarse de un registro exacto. El resumen no sustituye al hecho original. El uso de un modelo de lenguaje de 7.000 millones de parámetros en ese estudio no justifica introducirlo en el candidato local.

**Un recuerdo útil no es necesariamente el de mayor error.** [Mattar y Daw, 2018](https://www.princeton.edu/~ndaw/md18.pdf) proponen prioridad según ganancia de actualizar una política y necesidad de visitar el estado. [Antonov y Dayan, 2025](https://www.nature.com/articles/s41467-025-56731-y) amplían el análisis a incertidumbre y creencias. Son modelos de planificación. Su valor normativo no se convierte directamente en una fórmula barata para seleccionar episodios financieros. [Roscow y colaboradores, 2025](https://www.nature.com/articles/s41467-025-65354-2) aportan conducta de ratas, registros y modelización de replay asociado al error de predicción de recompensa. Ese error no equivale al MAE de un predictor pasivo. Se propone medir utilidad predictiva incremental del conjunto recuperado, frente a selección uniforme y por error.

**Conviene evaluar contexto, continuidad y confianza por separado.** [Baror y colaboradores, 2026](https://www.nature.com/articles/s41562-026-02403-w) encuentran efectos diferentes sobre continuidad y memoria según la manipulación contextual. No justifican borrar todo el estado cuando cambia un régimen estimado. [Koolschijn y colaboradores, 2026](https://www.nature.com/articles/s41467-026-70659-x) observan una extensión de asociaciones que también aumenta confusiones. Generalizar más no garantiza recordar mejor. [Davidson y colaboradores, 2025](https://www.eneuro.org/content/12/6/ENEURO.0124-25.2025) estudian señales EEG compartidas entre dos tareas perceptivas y confianza. No establecen un indicador universal de fiabilidad. Una cabeza de error debe validarse por separado de la puerta de recuperación.

**La habilidad se demuestra en una tarea delimitada.** [Kahneman y Klein, 2009](https://pubmed.ncbi.nlm.nih.gov/19739881/) relacionan intuición experta con regularidades aprendibles y oportunidades de feedback. Experiencia, confianza y acierto son propiedades diferentes. [Fernandez Velasco y colaboradores, 2025](https://doi.org/10.1073/pnas.2407814122), con su [corrección](https://pmc.ncbi.nlm.nih.gov/articles/PMC12130864/), estudian preparación anticipada de decisiones en 43 taxistas. [Van Opheusden y colaboradores, 2023](https://www.nature.com/articles/s41586-023-06124-2) infieren mayor profundidad de planificación con experiencia en un juego. No comparan con profesionales financieros. Cuatro lecturas de memoria no equivalen a cuatro pasos de razonamiento humano.

[Callaway y colaboradores, 2022](https://cocosci.princeton.edu/papers/callawayrationaluse.pdf) formalizan la asignación de cálculo mediante utilidad esperada y coste, con experimentos conductuales. La traducción comprobable sería aprender cuándo otra lectura mejora el error lo suficiente para compensar su coste. No se denomina «sénior» al modelo por su número de expertos. Se medirían calidad, calibración, uso de contexto, respuesta al feedback y coste. Una comparación humana necesitaría personas, acceso informativo equivalente y un protocolo independiente que todavía no existe.

## Qué exigen los antecedentes financieros

| Fuente | Comparación que evita una atribución incorrecta |
| --- | --- |
| [SFM, KDD 2017](https://doi.org/10.1145/3097983.3098117) | La descomposición temporal ya tiene antecedentes financieros. En esta ampliación se consultaron resumen y repositorio, sin atribuir lectura completa ni reproducción. |
| [TRA, KDD 2021](https://arxiv.org/html/2106.12950v2) | Enrutar según errores históricos no es una idea nueva. Su objetivo, horizonte y orden de actualización difieren del retorno residual diario. |
| [DoubleAdapt, KDD 2023](https://arxiv.org/abs/2306.09862v3) | La adaptación de datos y modelo requiere un control explícito frente a cambiar solo estado de memoria. Sus reajustes no son gratuitos ni equivalentes a parámetros congelados. |
| [MASTER, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/27767) | Contexto de mercado y relaciones entre activos deben compararse con agregaciones sencillas. Atención sobre todos los pares aumenta el coste con el cuadrado del universo. |
| [Financial RAG, v2 de junio de 2026](https://arxiv.org/html/2605.31201v2) | Es un antecedente cercano de lector congelado y recuperación adaptada con feedback maduro. No presentar esta combinación como una invención. La utilidad de una fuente no acredita su veracidad. |
| [TIEM, v5 de septiembre de 2026](https://arxiv.org/abs/2608.13024v5) | Combina evidencia y memoria financiera. En la evaluación principal leída la memoria permanece congelada. Eso no valida adaptación online de pesos en MARS-TITAN. |
| [When Does Context Routing Help?, v1 de agosto de 2026](https://arxiv.org/html/2608.25128v1) | Obliga a comparar contexto informativo y capacidad añadida. Usa una tarea de precios y noticias que no equivale al panel completo del proyecto. Su diagnóstico por información mutua tiene límites matemáticos explicados en el apéndice. |
| [Levy, 2026](https://doi.org/10.1111/1475-679x.70058) | Motiva controles de conocimiento retrospectivo y sensibilidad a perturbaciones numéricas. No convertir razonamiento verbal en aritmética fiable ni toda degradación al perturbar datos en prueba de memorización. |
| [When Agents Trade, preprint v2](https://arxiv.org/abs/2510.11695v2) | Aporta evaluación prospectiva de agentes con entradas sincronizadas. El estudio leído cubre dos acciones y dos criptomonedas durante dos meses. La revisión humana de noticias no es una comparación con operadores profesionales. |

La [auditoría de preentrenamiento](../data/pretraining-audit.md) y el [contrato de datos](../data/data-contract.md) siguen siendo necesarios. La cobertura multimodal de FinMultiTime no demuestra que cada dato fuera conocido en la fecha de decisión. El universo retrospectivo, la expansión de fundamentales por periodo y los gráficos de intervalos posteriores requieren comprobaciones propias. Una política temporal correcta no repara un codificador previamente expuesto al futuro.

## Variante ampliada y controles que la pueden descartar

### Lectura contextual con feedback verificable

Se mantiene un único codificador, la misma cabeza predictiva, K = 1 y el banco de episodios elegibles. La ruta base usa la lectura original. La contextual modifica la recuperación con información disponible de mercado, sector, publicaciones y estado filtrado. Ambas reciben las cuatro modalidades y macro. «Base» no significa retirar información de entrada a un competidor.

Las dos rutas calculan y registran sus predicciones antes de conocer la etiqueta. Al madurar, se obtiene una diferencia pareada de errores, agregada con la misma población de activos por sesión:

$$
u_t=L_t^{base}-L_t^{contexto},\qquad
L_t^j=\frac1{|S_t|}\sum_{i\in S_t}|y_{i,t}-\hat y_{i,t}^j|.
$$

La mezcla puede empezar por un peso constante elegido en desarrollo. Sus controles son una combinación adaptativa sin contexto y una puerta contextual pequeña. La puerta se ajusta con predicciones históricas generadas mediante cortes temporales internos, sin usar errores de ajuste como si fueran pronósticos fuera de muestra. Sus parámetros se congelan al evaluar. Si se estudian pesos online, su actualización se declara como estado explícito con feedback maduro y configuración fija.

Ambas predicciones existen, por lo que no hace falta tratar esta selección como un bandit. El precio observado no depende de qué pronóstico se publica. El coste de calcular las dos rutas se incluye. Si después se quiere calcular solo una, desaparece esa información completa y hará falta otro experimento, con costes y observación de errores distintos.

Hay un problema de atribución adicional. Si la escritura usa error predictivo, cambiar la salida cambia la memoria futura aunque se conserve la misma fórmula del selector. El primer diagnóstico de lectura ancla las escrituras a la predicción base, también calculada y registrada. Todas sus variantes comparten esa trayectoria de memoria. Así se aísla qué cambia al leer y combinar. Este control no estima el rendimiento de cada política con una historia de memoria propia.

Una segunda comparación, identificada como sistema completo, permitiría a cada variante escribir según su propia predicción. Tendría estados, cursores y checkpoints independientes y compararía trayectorias completas. La utilidad de una lectura no se reparte automáticamente entre cada noticia o episodio citado. La contribución individual necesita ablaciones adicionales, y su coste cuenta.

Se descarta la puerta si no supera la mezcla constante o la adaptación sin contexto a presupuesto comparable, si solo gana al aumentar capacidad, o si depende de segmentos elegidos tras ver su resultado. El contraste con contexto degradado se reserva a diagnósticos controlados. Desplazar datos rompe relaciones temporales y no es por sí solo un test válido de independencia.

### Escritura que distinga ruido y cambio

Se conserva el selector original con error maduro, anomalía y relevancia económica. La ampliación añade por separado una escala causal del error y una señal acumulada de persistencia. No se sustituye todo por una puntuación opaca de «sorpresa». Se mantiene el mismo cupo de escrituras, conjunto candidato, capacidad y cuotas uniforme y reciente.

Los controles sintéticos de desarrollo separan señal conocida, ausencia de señal, ruido de observación, cambio de dinámica, salto aislado y regreso a un contexto anterior. El generador conoce la verdad para evaluar, pero esos campos no entran en la observación del agente. La validación principal continúa siendo real. Se rechaza el mecanismo si retiene contaminación repetidamente, olvida contextos útiles o no supera uniformidad con el mismo trabajo.

Un HMM o detector de cambios aporta probabilidades estimadas, no etiquetas verdaderas de régimen. En evaluación se usa filtrado con el prefijo, nunca suavizado que consulte el futuro. La [revisión de Markov](markov-regimes.md) ya delimita esa diferencia. Ni el número de estados ni la definición de crisis se seleccionan con la evaluación final.

### Memoria asociativa estable como variante independiente

El [apéndice matemático](memory-mathematics.md) deriva condiciones de no expansión para la regla escalar y estudia una actualización proximal de cohorte. Esta última resuelve un sistema definido positivo y permite expresar cotas de condición y contracción bajo entradas fijas. La invariancia algebraica a permutar activos puede facilitar el procesamiento por lotes.

No conserva exactamente la dinámica de la regla explícita. Requiere otra configuración, comparación de calidad y presupuesto de cálculo. Debe competir con delta normalizada y, si el coste lo permite, RLS con olvido. Una factorización más estable puede costar más que una escritura de rango uno. La calidad del diseño se decidirá por error y coste observados, no solo por una cota favorable.

### Ampliaciones condicionadas

La compresión se estudiaría cuando el presupuesto de memoria sea un cuello de botella medido. El contraste sería banco exacto frente a prototipos más excepciones exactas, con los mismos bytes totales y referencias recuperables. Se medirían distorsión numérica, pérdida de episodios raros y MAE, además de similitud de embeddings. No se generarían cifras o noticias para reconstruir el pasado.

La consolidación por replay permanece dentro de entrenamiento o reajustes programados con pasado. Uniformidad, prioridad por error y utilidad contextual deben usar la misma cantidad de ejemplos y actualizaciones. El coste de aprender el selector se incluye. La destilación solo procede si el profesor mejora al alumno directo y se contabiliza la generación de sus objetivos.

La elección adaptativa de K se estudiaría después de K fijo en 1, 2 y 4. Una puerta puede estimar

$$
\widehat V_K(s)=\widehat{\mathbb E}[|y-\hat y_1|-|y-\hat y_K|\mid s]
-\lambda(c_K-c_1),
$$

con estado s disponible tras K = 1 y costes que incluyen la propia puerta. Se aprende en desarrollo con pronósticos temporales válidos. No ejecuta K = 4 para decidir retrospectivamente que era innecesario. La comparación incluye asignación aleatoria de igual cálculo medio y latencias de cola. Todos los casos admitidos siguen teniendo una predicción para el MAE principal. La abstención y su cobertura se evalúan aparte.

## Comparaciones justas y control del sobreajuste

La [evaluación temporal](protocol.md) gobierna también estas variantes. Se conserva la identidad de las muestras, sus exclusiones y las cuatro ventanas. Entrenamiento ajusta parámetros y transformaciones. Validación selecciona configuración y checkpoint. Calibración ajusta los intervalos de la política ya seleccionada. Evaluación estima su resultado fuera de selección. El test final de 2024 permanece cerrado. Si una evaluación se consulta para rediseñar el modelo, pasa a ser evidencia exploratoria y no puede seguir presentándose como confirmatoria.

| Eje | Regla de comparación |
| --- | --- |
| Datos e información | Mismas filas admitidas, modalidades, macro, objetivo, horizonte y calendario de disponibilidad. Una ablación diagnóstica se identifica aparte, sin ampliar su población gracias a menos requisitos. |
| Estado inicial | Mismo padre y estado inicial cuando el contraste lo permita. Reinicios, calentamiento, etiquetas pendientes y política de memoria declarados por ventana. No mezclar una ruta reiniciada con otra que conserva historia adicional. |
| Semillas | Semillas 42, 43 y 44, con corrientes separadas para inicialización, muestreo y simulación. Un mismo número de semilla no garantiza sorteos equivalentes entre algoritmos. Parear mundos y cohortes cuando comparten significado. |
| Búsqueda | Espacios, número de intentos, criterio de selección y presupuesto total fijados antes de evaluar. Incluir ajustes manuales, fallos y variantes descartadas. No seleccionar la mejor semilla. |
| Capacidad | Contar parámetros almacenados y activos, estados por activo, memoria externa, índices, optimizador, cola y duplicación de instantáneas. Igualar solo parámetros no iguala memoria ni información. |
| Cálculo | Mantener un contraste con presupuesto común y publicar además curvas de error frente a tiempo y memoria. Registrar muestras y actualizaciones reales, no comparar solo épocas. No se pueden igualar todos los recursos a la vez sin declarar cuál gobierna cada contraste. |
| Selección y parada | Validación temporal con la misma métrica, calendario, mejora mínima y paciencia predefinidos. Permitir seleccionar el padre inicial si una continuación empeora. Un máximo de iteraciones es una protección de presupuesto, no una prueba de convergencia. |
| Métricas | MAE residual por sesión sobre toda la población elegible, seguido de métricas secundarias y costes. La rentabilidad no selecciona al predictor. Una política financiera tiene su propio criterio y controles. |

La igualdad de actualizaciones merece un tratamiento específico. En las condiciones real más 25 % remuestreado y real más 25 % sintético, el contraste atribuye el efecto al tipo de ejemplos adicionales. Ambas necesitan la misma cantidad efectiva de actualizaciones y cada época debe recorrer todas las filas reales admitidas. Una parada independiente rompería ese emparejamiento. Se mantendría presupuesto fijo con selección del mejor estado, o una regla de parada conjunta fijada antes. Las comparaciones con parada independiente se informarían separadas, en la curva de calidad frente a coste. No se cambia por ello una campaña activa.

El early stopping reduce la exposición a deterioro observado en validación, pero no impide sobreajustarse a una validación consultada muchas veces. Hacen falta cortes internos para ajustar puertas y transformaciones y un tramo externo que no diseñe el modelo. La retención del mejor estado y la recuperación de interrupciones tienen funciones distintas. Se conservan solo los checkpoints previstos, con sustitución confirmada, y no se elimina el único estado recuperable antes de verificar el siguiente.

Las diferencias de MAE se comparan de forma pareada por sesión. Los intervalos deben mantener juntos los activos de una fecha y respetar dependencia temporal mediante bloques. La longitud de bloque, margen mínimo relevante y familia confirmatoria se fijan en desarrollo. No se cuentan ventanas solapadas como réplicas independientes ni tres semillas como tres historias económicas diferentes. Si se ensayan varios contrastes confirmatorios se declara el ajuste de multiplicidad y su familia. Los análisis por sectores o regímenes elegidos después son exploratorios.

Se mantienen referencias sencillas y el mejor competidor válido, aunque superen a la arquitectura propuesta. Un intervalo que admita beneficio y daño relevantes deja la comparación inconclusa. Las pérdidas de calibración y anchuras de intervalos se informan juntas. [Gibbs y Candès, 2024](https://www.jmlr.org/papers/v25/22-1218.html), §2.3 y §3, ofrecen adaptación de calibración con garantías bajo condiciones precisas. No equivale a cobertura de cada sector bajo cualquier deriva, especialmente después de seleccionar o abstenerse. La política completa necesita recalibración y comprobación propias.

En RL financiero, las políticas comparten observaciones, acciones, mundos, costes, restricciones, reglas de ejecución y predictores padre congelados. Se cuentan transiciones aprendidas, observadas y descartadas, actualizaciones y simulaciones de evaluación. El criterio financiero no es el MAE de un predictor que no cambia. Un historial con acciones distintas induce estados distintos, de modo que parear semillas no convierte todas las trayectorias en idénticas. La comparación sigue siendo simulada. La admisión de posiciones históricas persistentes requiere la procedencia acreditada de OHLC y acciones corporativas.

## Coste local y condiciones para pasar a implementación

El diseño base propone E = 8.192 episodios, 256 rasgos base, claves de 128 dimensiones y hasta ocho episodios únicos por lectura. Son valores de diseño. Solo rasgos y claves FP32 ocupan 12 MiB. Para 128 consultas, una matriz completa de puntuaciones ocupa 4 MiB y el producto denso requiere unos 268 millones de operaciones contando multiplicación y suma por separado. Faltan metadatos, top-k, proyecciones, escrituras, cola y copias. Dos rutas no son gratuitas aunque compartan el banco.

Una matriz asociativa 128 × 128 FP32 ocupa 64 KiB. Replicarla en 128 activos suma 8 MiB por capa antes de momento e instantáneas. Pesos, gradientes y dos momentos de Adam en FP32 suman 16 bytes por parámetro, sin activaciones. Estas cuentas son aritmética de tamaños, no medidas de RAM o VRAM del candidato.

La futura implementación conservaría C++20 nativo para tramos completos que lo justifiquen, bibliotecas matemáticas maduras y primitivas CUDA existentes. Búsqueda densa pequeña y matrices de Gram se expresarían con productos por lotes. Solo se añadirían índices aproximados, kernels o cachés cuando el perfil completo demostrase su utilidad y se verificasen recuperación y paridad. Comprimir claves, cambiar bloques o compartir estados puede alterar el modelo y debe declararse cuando ocurra.

Antes de decidir entre alternativas se medirían preparación, entrenamiento, lectura, escritura, consolidación y evaluación como recorrido conjunto. Se registrarían formas reales, hardware, versiones, precisión, calentamiento, repeticiones, dispersión, tiempo total, caudal, picos de RAM/VRAM y transferencias. Se incluyen generación de objetivos del profesor, ajuste de puertas y búsqueda. Sin instrumentación adecuada no se publican cifras de energía. La disponibilidad de 32 GiB de RAM y 8 GiB de VRAM no demuestra que cualquier combinación quepa.

La secuencia de contraste es acotada: reproducir la referencia y sus invariantes, evaluar la lectura contextual, después escritura y actualización asociativa, y estudiar una interacción pequeña solo si los componentes justifican su coste. La compresión y la elección adaptativa de K necesitan evidencia previa. No se propone una búsqueda cartesiana de todos los mecanismos revisados.

Las pruebas exigidas para una implementación incluyen modificar el sufijo futuro sin cambiar decisiones previas, retrasar publicaciones, permutar activos de una sesión, deduplicar recuerdos y recuperar estado con el mismo cursor, RNG y etiquetas pendientes. Los mundos sin señal deben impedir atribuir al método conocimiento que procede del generador o de la evaluación. Un fallo temporal o de recuperación invalida una comparación aunque su métrica sea favorable.

El trabajo ejecutado en esta revisión se limita a fuentes, diseño y [comprobaciones algebraicas reproducibles](memory-mathematics.md). La posible aportación futura es medir cuándo estos mecanismos conservan señal útil y cuándo fallan en un panel multimodal temporalmente válido. No hay todavía una mejora del candidato, un resultado frente a profesionales ni una nueva arquitectura validada.
